import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from skimage import exposure
from matplotlib.colors import LinearSegmentedColormap

def visualize_enhanced_samples(base_path: str, dataset: str, num_samples: int = 3):
    """
    Visualize sample slices with enhanced contrast and better mask visualization

    Args:
        base_path (str): Base path to the dataset
        dataset (str): Name of the dataset to visualize
        num_samples (int): Number of samples to visualize
    """
    # Construct paths
    dataset_path = Path(base_path) / 'train' / dataset
    images_path = dataset_path / 'images'
    labels_path = dataset_path / 'labels'

    # Check if paths exist
    if not images_path.exists() or not labels_path.exists():
        print(f"Dataset {dataset} not found or incomplete")
        return

    # Get sorted list of files
    image_files = sorted(list(images_path.glob('*.tif')))
    label_files = sorted(list(labels_path.glob('*.tif')))

    # Calculate step size to get evenly spaced samples
    step = len(image_files) // num_samples
    indices = list(range(0, len(image_files), step))[:num_samples]

    # Create custom colormap for masks
    colors = ['black', 'red']  # Black background, red vessels
    vessel_cmap = LinearSegmentedColormap.from_list('vessel_map', colors)

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    #fig.suptitle(f'Enhanced Visualization of {dataset}', fontsize=16)

    for idx, i in enumerate(indices):
        # Load image and mask
        image = tifffile.imread(str(image_files[i]))
        mask = tifffile.imread(str(label_files[i]))

        # Enhance image contrast
        p2, p98 = np.percentile(image, (2, 98))
        image_enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))

        # Create color overlay
        overlay = np.zeros((*image.shape, 4))  # RGBA
        overlay[..., 0] = mask  # Red channel
        overlay[..., 3] = mask * 0.7  # Alpha channel (transparency)

        # Plot original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original - {image_files[i].name}')
        axes[idx, 0].axis('off')

        # Plot enhanced image
        axes[idx, 1].imshow(image_enhanced, cmap='gray')
        axes[idx, 1].set_title('Enhanced')
        axes[idx, 1].axis('off')

        # Plot mask overlay
        axes[idx, 2].imshow(image_enhanced, cmap='gray')
        axes[idx, 2].imshow(mask, cmap=vessel_cmap, alpha=0.7)
        axes[idx, 2].set_title('Mask Overlay')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Kaggle dataset path
    kaggle_path = '/kaggle/input/blood-vessel-segmentation'

    # List of datasets
    datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_dense', 'kidney_3_sparse']

    # Visualize samples from each dataset
    for dataset in datasets:
        print(f"\nVisualizing {dataset}...")
        visualize_enhanced_samples(kaggle_path, dataset)

        # Add a pause between datasets to make it easier to view
        input("Press Enter to continue to next dataset...")

if __name__ == "__main__":
    main()
