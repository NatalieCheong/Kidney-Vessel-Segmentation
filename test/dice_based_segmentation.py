import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import gc
from vessel_data_preprocessing import load_and_recreate_dataloaders
from unet_model_architecture import ResNetUNet

def segment_and_visualize(model, image_tensor, device):
    """
    Enhanced segmentation visualization with probability map
    """
    model.eval()
    with torch.no_grad():
        # Move image to device and get prediction
        image = image_tensor.to(device)
        output = model(image.unsqueeze(0))
        prediction = (output > 0.5).float()
        probability = output.cpu().squeeze().numpy()  # Get raw probability

        # Convert to numpy for visualization
        original = image.cpu().squeeze().numpy()
        segmentation = prediction.cpu().squeeze().numpy()

        # Create overlay with alpha based on probability
        overlay = np.zeros((*original.shape, 3))
        overlay[..., 0] = original  # Gray channel
        overlay[..., 1] = original  # Gray channel
        overlay[..., 2] = original  # Gray channel

        # Create probability-weighted mask (red for vessels, more intense = higher probability)
        prob_mask = np.zeros((*original.shape, 4))  # RGBA
        prob_mask[..., 0] = 1.0  # Red channel
        prob_mask[..., 3] = probability  # Alpha channel based on probability

        # Add red highlight for segmented vessels
        mask_region = segmentation > 0
        overlay[mask_region, 0] = 1.0  # Red channel
        overlay[mask_region, 1] = 0.0  # Green channel
        overlay[mask_region, 2] = 0.0  # Blue channel

        return original, segmentation, probability, overlay

def perform_vessel_segmentation(model, dataset_name, val_loader, device, num_samples=3):
    """
    Perform and visualize vessel segmentation with probability maps
    """
    # Get random batch
    dataiter = iter(val_loader)
    images, ground_truth = next(dataiter)

    # Select random indices
    batch_size = images.size(0)
    indices = random.sample(range(batch_size), min(num_samples, batch_size))

    # Create figure
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5*num_samples))
    fig.suptitle(f'Enhanced Blood Vessel Segmentation Results for {dataset_name}', fontsize=16)

    for idx, sample_idx in enumerate(indices):
        image = images[sample_idx]
        gt_mask = ground_truth[sample_idx]

        # Get segmentation results
        original, segmentation, probability, overlay = segment_and_visualize(model, image, device)

        # Display results
        # Original image
        axes[idx, 0].imshow(original, cmap='gray')
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # Ground truth
        axes[idx, 1].imshow(gt_mask.squeeze().cpu().numpy(), cmap='Reds')
        axes[idx, 1].set_title('Ground Truth Vessels')
        axes[idx, 1].axis('off')

        # Probability map
        axes[idx, 2].imshow(probability, cmap='hot')
        axes[idx, 2].set_title('Probability Map')
        axes[idx, 2].axis('off')

        # Segmented vessels
        axes[idx, 3].imshow(segmentation, cmap='Reds')
        axes[idx, 3].set_title('Segmented Vessels')
        axes[idx, 3].axis('off')

        # Vessel overlay
        axes[idx, 4].imshow(overlay)
        axes[idx, 4].set_title('Vessel Overlay')
        axes[idx, 4].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    plt.close()

def visualization_main():
    """
    Main function to perform enhanced vessel segmentation visualization on all datasets
    """
    try:
        print("Starting enhanced vessel segmentation visualization...")

        # Load dataloaders
        print("\nLoading dataloaders...")
        dataloader_path = Path('/kaggle/working/dataloader_info')
        dataloaders = load_and_recreate_dataloaders(dataloader_path)

        # Initialize model
        print("\nInitializing model...")
        model = ResNetUNet(n_classes=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Find the latest training directory with a more flexible pattern
        training_dirs = list(Path('/kaggle/working').glob('*training_results_*'))
        if not training_dirs:
            raise ValueError("No training results directories found. Make sure you've run training first.")

        # Sort by modification time to get the latest
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        print(f"\nUsing models from: {latest_dir}")

        # Process each dataset
        for dataset_name, loaders in dataloaders.items():
            print(f"\nProcessing {dataset_name}")

            # Check for regular model
            model_path = latest_dir / dataset_name / 'best_model.pth'

            print(f"Loading model from: {model_path}")

            if model_path.exists():
                try:
                    # Load model weights
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model with Best Dice: {checkpoint['best_dice']:.4f}")

                    # Perform enhanced segmentation
                    print("Performing enhanced vessel segmentation...")
                    perform_vessel_segmentation(
                        model=model,
                        dataset_name=dataset_name,
                        val_loader=loaders['val'],
                        device=device
                    )
                except Exception as e:
                    print(f"Error processing {dataset_name}: {str(e)}")
            else:
                print(f"No model found at {model_path}")

            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nEnhanced vessel segmentation visualization completed!")

    except Exception as e:
        print(f"Error in enhanced vessel segmentation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualization_main()
