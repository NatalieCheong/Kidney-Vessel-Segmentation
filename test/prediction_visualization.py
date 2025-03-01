import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import gc

def visualize_predictions(model, dataset_name, val_loader, device, num_samples=3):
    """
    Visualize model predictions for a given dataset
    """
    model.eval()

    # Get random batch
    try:
        dataiter = iter(val_loader)
        images, masks = next(dataiter)

        # Select random indices
        batch_size = images.size(0)
        indices = random.sample(range(batch_size), min(num_samples, batch_size))

        # Create figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        fig.suptitle(f'Model Predictions for {dataset_name}', fontsize=16)

        with torch.no_grad():
            # Move to device
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            outputs = model(images)
            predictions = (outputs > 0.5).float()

            # Display images
            for idx, sample_idx in enumerate(indices):
                # Get single sample
                image = images[sample_idx].cpu().squeeze().numpy()
                mask = masks[sample_idx].cpu().squeeze().numpy()
                pred = predictions[sample_idx].cpu().squeeze().numpy()

                # Original image
                axes[idx, 0].imshow(image, cmap='gray')
                axes[idx, 0].set_title('Original Image')
                axes[idx, 0].axis('off')

                # Ground truth mask
                axes[idx, 1].imshow(mask, cmap='Reds')
                axes[idx, 1].set_title('Ground Truth')
                axes[idx, 1].axis('off')

                # Prediction
                axes[idx, 2].imshow(pred, cmap='Reds')
                axes[idx, 2].set_title('Prediction')
                axes[idx, 2].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()

        # Clear memory
        del images, masks, outputs, predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error visualizing predictions for {dataset_name}: {str(e)}")

def main():
    """
    Main function to visualize predictions for all datasets
    """
    try:
        print("Starting visualization...")

        # Load dataloaders
        print("\nLoading dataloaders...")
        dataloader_path = Path('/kaggle/working/dataloader_info')
        dataloaders = load_and_recreate_dataloaders(dataloader_path)

        # Initialize model
        print("\nInitializing model...")
        model = ResNetUNet(n_classes=1)  # Using original ResNetUNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Find the latest training directory with a more flexible pattern
        training_dirs = list(Path('/kaggle/working').glob('*training_results_*'))
        if not training_dirs:
            raise ValueError("No training results directories found. Make sure you've run training first.")

        # Sort by modification time to get the latest
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        print(f"\nUsing results from: {latest_dir}")

        # Visualize for each dataset
        for dataset_name, loaders in dataloaders.items():
            print(f"\nVisualizing predictions for {dataset_name}")

            # Load best model for this dataset
            model_path = latest_dir / dataset_name / 'best_model.pth'
            print(f"Looking for model at: {model_path}")

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model with Best Dice: {checkpoint['best_dice']:.4f}")

                    # Visualize predictions
                    visualize_predictions(
                        model=model,
                        dataset_name=dataset_name,
                        val_loader=loaders['val'],
                        device=device
                    )
                except Exception as e:
                    print(f"Error loading model for {dataset_name}: {str(e)}")
            else:
                print(f"No model found at {model_path}")

        # Clear memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
