import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import gc
import traceback
from vessel_data_preprocessing import load_and_recreate_dataloaders
from unet_model_architecture import ResNetUNet
from enhanced_vessel_segmentation import segment_and_visualize

def visualize_using_surface_dice():
    """
    Visualization function that uses models with best Surface Dice instead of best Dice
    """
    try:
        print("Starting vessel segmentation based on best Surface Dice...")

        # Load dataloaders
        print("\nLoading dataloaders...")
        dataloader_path = Path('/kaggle/working/dataloader_info')
        dataloaders = load_and_recreate_dataloaders(dataloader_path)

        # Initialize model
        print("\nInitializing model...")
        model = ResNetUNet(n_classes=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Find the latest training directory
        training_dirs = list(Path('/kaggle/working').glob('*training_results_*'))
        if not training_dirs:
            raise ValueError("No training results directories found. Make sure you've run training first.")

        # Sort by modification time to get the latest
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        print(f"\nUsing models from: {latest_dir}")

        # Process each dataset
        for dataset_name, loaders in dataloaders.items():
            print(f"\nProcessing {dataset_name}")

            # Check for best model
            model_path = latest_dir / dataset_name / 'best_model.pth'

            if model_path.exists():
                try:
                    # Load checkpoint and check metrics
                    checkpoint = torch.load(model_path, map_location=device)
                    best_dice = checkpoint['best_dice']
                    best_surface_dice = checkpoint['best_surface_dice']

                    print(f"Found model with Best Dice: {best_dice:.4f} and Best Surface Dice: {best_surface_dice:.4f}")

                    # Load model weights
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # Perform enhanced segmentation
                    print("Performing vessel segmentation using best Surface Dice model...")

                    # Create figure title highlighting Surface Dice
                    title = f'Surface Dice Optimized Vessel Segmentation for {dataset_name} (SD: {best_surface_dice:.4f})'

                    # Get random batch
                    dataiter = iter(loaders['val'])
                    images, ground_truth = next(dataiter)

                    # Select random indices
                    batch_size = images.size(0)
                    num_samples = 3
                    indices = random.sample(range(batch_size), min(num_samples, batch_size))

                    # Create figure
                    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
                    fig.suptitle(title, fontsize=16)

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
                        axes[idx, 2].set_title('Vessel Probability')
                        axes[idx, 2].axis('off')

                        # Overlay
                        axes[idx, 3].imshow(overlay)
                        axes[idx, 3].set_title('Vessel Overlay')
                        axes[idx, 3].axis('off')

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    plt.show()
                    plt.close()

                except Exception as e:
                    print(f"Error processing {dataset_name}: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"No model found at {model_path}")

            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nSurface Dice based vessel segmentation completed!")

    except Exception as e:
        print(f"Error in vessel segmentation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_using_surface_dice()
