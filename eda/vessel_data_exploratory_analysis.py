import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import tifffile
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class KidneyVesselEDA:
    def __init__(self, base_path: str):
        """
        Initialize the EDA class

        Args:
            base_path (str): Path to the Kaggle dataset directory
        """
        self.base_path = Path(base_path)
        self.train_path = self.base_path / 'train'
        self.datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2',
                        'kidney_3_dense', 'kidney_3_sparse']

    def load_dataset_info(self) -> Dict:
        """
        Load basic information about each dataset

        Returns:
            Dict: Dictionary containing dataset statistics
        """
        dataset_info = {}

        for dataset in self.datasets:
            dataset_path = self.train_path / dataset
            if dataset_path.exists():
                images_path = dataset_path / 'images'
                labels_path = dataset_path / 'labels'

                if images_path.exists():
                    n_images = len(list(images_path.glob('*.tif')))
                else:
                    n_images = 0

                if labels_path.exists():
                    n_labels = len(list(labels_path.glob('*.tif')))
                else:
                    n_labels = 0

                # Get image dimensions from first image
                if n_images > 0:
                    first_image = tifffile.imread(str(next(images_path.glob('*.tif'))))
                    dimensions = first_image.shape
                else:
                    dimensions = None

                dataset_info[dataset] = {
                    'n_images': n_images,
                    'n_labels': n_labels,
                    'dimensions': dimensions
                }

        return dataset_info

    def analyze_class_distribution(self, dataset: str) -> Tuple[float, float]:
        """
        Analyze the class distribution (vessel vs non-vessel) in a dataset

        Args:
            dataset (str): Name of the dataset to analyze

        Returns:
            Tuple[float, float]: Percentage of vessel and non-vessel pixels
        """
        labels_path = self.train_path / dataset / 'labels'
        if not labels_path.exists():
            return None

        total_pixels = 0
        vessel_pixels = 0

        for label_file in tqdm(list(labels_path.glob('*.tif')), desc=f'Analyzing {dataset}'):
            mask = tifffile.imread(str(label_file))
            total_pixels += mask.size
            vessel_pixels += np.sum(mask > 0)

        vessel_percentage = (vessel_pixels / total_pixels) * 100
        non_vessel_percentage = 100 - vessel_percentage

        return vessel_percentage, non_vessel_percentage

    def visualize_sample_slices(self, dataset: str, n_samples: int = 5) -> None:
        """
        Visualize sample slices from a dataset with their corresponding masks

        Args:
            dataset (str): Name of the dataset to visualize
            n_samples (int): Number of samples to visualize
        """
        images_path = self.train_path / dataset / 'images'
        labels_path = self.train_path / dataset / 'labels'

        if not images_path.exists() or not labels_path.exists():
            print(f"Dataset {dataset} not found or incomplete")
            return

        image_files = sorted(list(images_path.glob('*.tif')))
        label_files = sorted(list(labels_path.glob('*.tif')))

        # Select evenly spaced samples
        indices = np.linspace(0, len(image_files)-1, n_samples, dtype=int)

        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3*n_samples))
        fig.suptitle(f'Sample Slices from {dataset}')

        for idx, (ax_row, i) in enumerate(zip(axes, indices)):
            # Load image and mask
            image = tifffile.imread(str(image_files[i]))
            mask = tifffile.imread(str(label_files[i]))

            # Display image
            ax_row[0].imshow(image, cmap='gray')
            ax_row[0].set_title(f'Slice {str(image_files[i]).split("/")[-1]}')
            ax_row[0].axis('off')

            # Display mask
            ax_row[1].imshow(mask, cmap='binary')
            ax_row[1].set_title(f'Mask {str(label_files[i]).split("/")[-1]}')
            ax_row[1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_class_distribution(self, distribution_data: Dict) -> None:
        """
        Plot the class distribution across datasets

        Args:
            distribution_data (Dict): Dictionary containing class distribution data
        """
        datasets = list(distribution_data.keys())
        vessel_percentages = [d[0] for d in distribution_data.values()]
        non_vessel_percentages = [d[1] for d in distribution_data.values()]

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35

        ax.bar(datasets, vessel_percentages, width, label='Vessel')
        ax.bar(datasets, non_vessel_percentages, width, bottom=vessel_percentages, label='Non-vessel')

        ax.set_ylabel('Percentage')
        ax.set_title('Class Distribution Across Datasets')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
def main():
    # Initialize EDA class with Kaggle dataset path
    kaggle_path = '/kaggle/input/blood-vessel-segmentation'
    eda = KidneyVesselEDA(kaggle_path)

    # Get dataset information
    print("Analyzing dataset information...")
    dataset_info = eda.load_dataset_info()

    # Print dataset statistics
    print("\nDataset Statistics:")
    for dataset, info in dataset_info.items():
        print(f"\n{dataset}:")
        print(f"  Number of images: {info['n_images']}")
        print(f"  Number of labels: {info['n_labels']}")
        print(f"  Image dimensions: {info['dimensions']}")

    # Analyze class distribution for training datasets
    print("\nAnalyzing class distribution...")
    distribution_data = {}
    for dataset in eda.datasets:
        if dataset_info[dataset]['n_labels'] > 0:
            distribution = eda.analyze_class_distribution(dataset)
            if distribution:
                distribution_data[dataset] = distribution
                print(f"\n{dataset}:")
                print(f"  Vessel pixels: {distribution[0]:.2f}%")
                print(f"  Non-vessel pixels: {distribution[1]:.2f}%")

    # Plot class distribution
    eda.plot_class_distribution(distribution_data)

    # Visualize sample slices from each dataset
    print("\nVisualizing sample slices...")
    for dataset in eda.datasets:
        if dataset_info[dataset]['n_labels'] > 0:
            print(f"\nVisualizing {dataset}...")
            eda.visualize_sample_slices(dataset)

if __name__ == "__main__":
    main()
