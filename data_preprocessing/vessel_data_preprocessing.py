import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy.ndimage import binary_dilation, binary_closing
import albumentations as A
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings("ignore")


def clear_memory():
    """Aggressively clear memory"""
    plt.close('all')
    gc.collect()

class_weights = {
    'kidney_1_dense': {'dice_weight': 0.7, 'bce_weight': 0.3},
    'kidney_1_voi': {'dice_weight': 0.6, 'bce_weight': 0.4},
    'kidney_2': {'dice_weight': 0.7, 'bce_weight': 0.3},
    'kidney_3_sparse': {'dice_weight': 0.8, 'bce_weight': 0.2}
}

class VesselDataset(Dataset):
    """
    Dataset class for vessel segmentation data
    """
    def __init__(self,
                 image_files: List[Path],
                 label_files: List[Path],
                 transform: Optional[A.Compose] = None,
                 size: Tuple[int, int] = (512, 512)):
        """
        Initialize the dataset

        Args:
            image_files (List[Path]): List of paths to image files
            label_files (List[Path]): List of paths to label files
            transform (Optional[A.Compose]): Albumentations transforms to apply
            size (Tuple[int, int]): Target size for resizing (height, width)
        """
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform
        self.target_h, self.target_w = size

        # Validate files
        self._validate_files()

        # Get original dimensions from first image
        try:
            first_image = tifffile.imread(str(image_files[0]))
            orig_h, orig_w = first_image.shape
            print(f"Dataset: Original dimensions: {orig_h}x{orig_w}, Target dimensions: {self.target_h}x{self.target_w}")

            # Clear memory
            del first_image
            gc.collect()

        except Exception as e:
            print(f"Warning: Could not read first image: {str(e)}")

    def _validate_files(self):
        """Validate that all files exist and match"""
        if len(self.image_files) != len(self.label_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) "
                           f"!= number of labels ({len(self.label_files)})")

        # Check all files exist
        for img_path, label_path in zip(self.image_files, self.label_files):
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")

    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.image_files)

    def preprocess_image(self,
                        image: np.ndarray,
                        is_mask: bool = False) -> np.ndarray:
        """
        Preprocess image to target size

        Args:
            image (np.ndarray): Input image or mask
            is_mask (bool): Whether the input is a mask

        Returns:
            np.ndarray: Preprocessed image or mask
        """
        try:
            if is_mask:
                # Use nearest neighbor for masks to preserve binary values
                processed = cv2.resize(
                    image.astype(np.uint8),
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                # Use bilinear interpolation for images
                processed = cv2.resize(
                    image.astype(np.float32),
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_LINEAR
                )
            return processed

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            # Return zero array of correct shape and type
            return np.zeros((self.target_h, self.target_w),
                          dtype=np.uint8 if is_mask else np.float32)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0,1] range with safe division

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Normalized image
        """
        try:
            image_min = image.min()
            image_max = image.max()

            if image_max - image_min == 0:
                return np.zeros_like(image, dtype=np.float32)

            return (image - image_min) / (image_max - image_min)

        except Exception as e:
            print(f"Error in normalization: {str(e)}")
            return np.zeros_like(image, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and mask tensors
        """
        try:
            # Load image and mask with memory management
            image = tifffile.imread(str(self.image_files[idx]))
            mask = tifffile.imread(str(self.label_files[idx]))

            # Basic input validation
            if image is None or mask is None:
                raise ValueError("Failed to load image or mask")

            if image.size == 0 or mask.size == 0:
                raise ValueError("Empty image or mask")

            # Normalize image
            image = self.normalize_image(image)

            # Convert mask to binary
            mask = (mask > 0).astype(np.uint8)

            # Resize both to target size
            image = self.preprocess_image(image, is_mask=False)
            mask = self.preprocess_image(mask, is_mask=True)

            # Apply transforms if specified
            if self.transform:
                transformed = self.transform(
                    image=image.astype(np.float32),
                    mask=mask
                )
                image = transformed['image']
                mask = transformed['mask']

            # Convert to tensors
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

            # Validate output tensors
            if torch.isnan(image).any() or torch.isnan(mask).any():
                raise ValueError("NaN values in output tensors")

            # Clear memory
            gc.collect()

            return image, mask

        except Exception as e:
            print(f"Error loading sample {idx} from {self.image_files[idx]}: {str(e)}")
            # Return zero tensors in case of error
            return (torch.zeros((1, self.target_h, self.target_w), dtype=torch.float32),
                   torch.zeros((1, self.target_h, self.target_w), dtype=torch.float32))

    def get_class_weights(self) -> Tuple[float, float]:
        """
        Calculate class weights based on the full dataset

        Returns:
            Tuple[float, float]: Weights for background and vessel classes
        """
        try:
            total_pixels = 0
            vessel_pixels = 0

            for label_file in self.label_files:
                mask = tifffile.imread(str(label_file))
                total_pixels += mask.size
                vessel_pixels += np.sum(mask > 0)

            background_pixels = total_pixels - vessel_pixels

            # Calculate weights (inverse frequency)
            background_weight = 1.0
            vessel_weight = (background_pixels / vessel_pixels) if vessel_pixels > 0 else 1.0

            return background_weight, vessel_weight

        except Exception as e:
            print(f"Error calculating class weights: {str(e)}")
            return 1.0, 1.0

def get_train_transforms(dataset_name):
    """Get improved but safer augmentation transforms"""
    # Base transforms for all datasets
    base_transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=30,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        )
    ]

    # Dataset-specific transforms (simplified)
    if 'sparse' in dataset_name:
        base_transforms.extend([
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.RandomBrightnessContrast(p=0.3)
        ])

    if 'dense' in dataset_name:
        base_transforms.extend([
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(clip_limit=2, p=0.3)
        ])

    # Add normalization as final transform
    base_transforms.append(
        A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0, p=1.0)
    )

    return A.Compose(base_transforms)

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
    ])

def process_dataset(kaggle_path: Path, dataset_name: str, target_size=(512, 512)) -> Tuple[DataLoader, DataLoader]:
    """Process dataset and create train/val splits"""
    print(f"\nProcessing {dataset_name}")

    try:
        # Setup paths
        images_path = kaggle_path / 'train' / dataset_name / 'images'
        labels_path = kaggle_path / 'train' / dataset_name / 'labels'

        if not images_path.exists() or not labels_path.exists():
            print(f"Required directories not found for {dataset_name}")
            return None, None

        # Get files and match them
        image_files = sorted(list(images_path.glob('*.tif')))
        label_files = []

        # Match files by index
        for img_file in image_files:
            img_idx = int(img_file.stem)
            matching_label = labels_path / f"{img_idx:04d}.tif"
            if matching_label.exists():
                label_files.append(matching_label)

        # Keep only images with matching labels
        image_files = image_files[:len(label_files)]

        if len(image_files) == 0:
            print(f"No matching pairs found for {dataset_name}")
            return None, None

        print(f"Found {len(image_files)} matching pairs")

        # Split into train and validation
        train_images, val_images, train_labels, val_labels = train_test_split(
            image_files, label_files,
            test_size=0.2,
            random_state=42
        )

        # Create datasets with fixed size
        train_dataset = VesselDataset(
            train_images,
            train_labels,
            transform=get_train_transforms(dataset_name),
            size=target_size
        )

        val_dataset = VesselDataset(
            val_images,
            val_labels,
            transform=get_val_transforms(),
            size=target_size
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"Created train dataloader with {len(train_dataset)} samples")
        print(f"Created val dataloader with {len(val_dataset)} samples")

        return train_loader, val_loader

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        gc.collect()
        return None, None

def create_overlay(image, mask, alpha=0.5):
    """Create overlay of mask on image"""
    # Ensure proper types
    image = image.astype(np.float32)
    mask = mask.astype(bool)

    # Create RGB version of grayscale image
    rgb_image = np.stack([image] * 3, axis=-1)

    # Create red mask overlay
    red_mask = np.zeros_like(rgb_image)
    red_mask[mask] = [1, 0, 0]  # Red color for mask

    # Combine image and mask
    overlay = (1 - alpha) * rgb_image + alpha * red_mask

    # Ensure values are in valid range
    overlay = np.clip(overlay, 0, 1)

    return overlay

def visualize_processed_dataset(images_path: Path, labels_path: Path, dataset_name: str, num_samples: int = 3):
    """Visualize samples from dataset including augmentations"""
    try:
        image_files = sorted(list(images_path.glob('*.tif')))
        label_files = sorted(list(labels_path.glob('*.tif')))

        if len(image_files) == 0 or len(label_files) == 0:
            print(f"No images or masks found for {dataset_name}")
            return

        # Select evenly spaced samples
        indices = np.linspace(0, len(image_files)-1, num_samples, dtype=int)

        # Get transforms
        train_transform = get_train_transforms(dataset_name)

        for idx in indices:
            # Load images with memory management
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Load and preprocess image
                image = tifffile.imread(str(image_files[idx]))
                image = (image - image.min()) / (image.max() - image.min())

                # Load and preprocess mask - keep as uint8
                mask = tifffile.imread(str(label_files[idx]))
                mask = (mask > 0).astype(np.uint8)  # Convert to uint8 instead of bool

                # Apply augmentation
                try:
                    augmented = train_transform(image=image.astype(np.float32),
                                             mask=mask)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                except Exception as e:
                    print(f"Augmentation error: {str(e)}")
                    continue

                # Create figure with two rows
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{dataset_name} - Sample {idx}')

                # Original images row
                axes[0, 0].imshow(image, cmap='gray')
                axes[0, 0].set_title('Original')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(mask, cmap='Reds')
                axes[0, 1].set_title('Original Mask')
                axes[0, 1].axis('off')

                # Create and show original overlay
                overlay = create_overlay(image, mask)
                axes[0, 2].imshow(overlay)
                axes[0, 2].set_title('Original Overlay')
                axes[0, 2].axis('off')

                # Augmented images row
                axes[1, 0].imshow(aug_image, cmap='gray')
                axes[1, 0].set_title('Augmented')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(aug_mask, cmap='Reds')
                axes[1, 1].set_title('Augmented Mask')
                axes[1, 1].axis('off')

                # Create and show augmented overlay
                aug_overlay = create_overlay(aug_image, aug_mask.astype(bool))
                axes[1, 2].imshow(aug_overlay)
                axes[1, 2].set_title('Augmented Overlay')
                axes[1, 2].axis('off')

                plt.tight_layout()
                plt.show()
                plt.close()

            # Clear memory after each sample
            gc.collect()

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        gc.collect()

def save_dataloader_info(dataloaders, save_dir):
    """
    Save dataloader configurations and splits for each dataset
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {}

    for dataset_name, loaders in dataloaders.items():
        # Get file paths from datasets
        train_dataset = loaders['train'].dataset
        val_dataset = loaders['val'].dataset

        dataset_info[dataset_name] = {
            'train': {
                'image_files': [str(path) for path in train_dataset.image_files],
                'label_files': [str(path) for path in train_dataset.label_files],
            },
            'val': {
                'image_files': [str(path) for path in val_dataset.image_files],
                'label_files': [str(path) for path in val_dataset.label_files],
            },
            'config': {
                'batch_size': loaders['train'].batch_size,
                'num_workers': loaders['train'].num_workers,
                'pin_memory': loaders['train'].pin_memory,
            }
        }

    # Save to JSON file
    with open(save_dir / 'dataloader_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"\nDataloader information saved to {save_dir / 'dataloader_info.json'}")

def load_and_recreate_dataloaders(save_dir):
    """
    Recreate dataloaders from saved information
    """
    save_dir = Path(save_dir)

    # Load saved information
    with open(save_dir / 'dataloader_info.json', 'r') as f:
        dataset_info = json.load(f)

    dataloaders = {}

    for dataset_name, info in dataset_info.items():
        # Create train dataset
        train_dataset = VesselDataset(
            image_files=[Path(p) for p in info['train']['image_files']],
            label_files=[Path(p) for p in info['train']['label_files']],
            transform=get_train_transforms(dataset_name)
        )

        # Create val dataset
        val_dataset = VesselDataset(
            image_files=[Path(p) for p in info['val']['image_files']],
            label_files=[Path(p) for p in info['val']['label_files']],
            transform=get_val_transforms()
        )

        # Create dataloaders
        config = info['config']
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        dataloaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader
        }

    print("\nDataloaders recreated successfully!")
    return dataloaders

def verify_saved_dataloaders(original_loaders, recreated_loaders):
    """Verify that recreated dataloaders match the original ones"""
    print("\nVerifying recreated dataloaders:")
    print("=" * 50)

    for dataset_name in original_loaders.keys():
        print(f"\nDataset: {dataset_name}")

        orig_train = original_loaders[dataset_name]['train']
        new_train = recreated_loaders[dataset_name]['train']

        orig_val = original_loaders[dataset_name]['val']
        new_val = recreated_loaders[dataset_name]['val']

        print(f"Train samples - Original: {len(orig_train.dataset)}, Recreated: {len(new_train.dataset)}")
        print(f"Val samples - Original: {len(orig_val.dataset)}, Recreated: {len(new_val.dataset)}")

        # Verify first batch shapes
        try:
            orig_batch = next(iter(orig_train))
            new_batch = next(iter(new_train))

            print("Batch shapes:")
            print(f"Original - Images: {orig_batch[0].shape}, Masks: {orig_batch[1].shape}")
            print(f"Recreated - Images: {new_batch[0].shape}, Masks: {new_batch[1].shape}")
            print("Shapes match:",
                  orig_batch[0].shape == new_batch[0].shape and
                  orig_batch[1].shape == new_batch[1].shape)

        except Exception as e:
            print(f"Error checking batch shapes: {str(e)}")
            continue

def save_dataloaders_main(dataloaders):
    """Save and verify dataloaders"""
    # Save directory
    save_dir = Path('/kaggle/working/dataloader_info')

    # Save dataloader information
    save_dataloader_info(dataloaders, save_dir)

    # Recreate dataloaders to verify
    recreated_loaders = load_and_recreate_dataloaders(save_dir)

    # Verify recreated dataloaders
    verify_saved_dataloaders(dataloaders, recreated_loaders)

def main():
    print("Starting Preprocessing Pipeline")
    print("="*50)

    # Declare global variables
    global train_loader, val_loader
    train_loader, val_loader = None, None

    # Setup paths
    kaggle_path = Path('/kaggle/input/blood-vessel-segmentation')
    save_dir = Path('/kaggle/working/dataloader_info')

    # List of datasets
    datasets = [
        'kidney_1_dense',
        'kidney_1_voi',
        'kidney_2',
        'kidney_3_dense',
        'kidney_3_sparse'
    ]

    # Process each dataset
    dataloaders = {}
    processed_datasets = []

    for dataset_name in datasets:
        try:
            print(f"\nProcessing {dataset_name}")
            print("-" * 30)

            # Process dataset with memory management
            train_loader, val_loader = process_dataset(
                kaggle_path,
                dataset_name
            )

            if train_loader and val_loader:
                dataloaders[dataset_name] = {
                    'train': train_loader,
                    'val': val_loader
                }
                processed_datasets.append(dataset_name)

                # Clear memory after successful processing
                gc.collect()

                # Visualize samples
                print(f"\nDisplaying samples from {dataset_name}")
                images_path = kaggle_path / 'train' / dataset_name / 'images'
                labels_path = kaggle_path / 'train' / dataset_name / 'labels'

                if images_path.exists() and labels_path.exists():
                    visualize_processed_dataset(images_path, labels_path, dataset_name)
                    print(f"Visualization completed for {dataset_name}")
                else:
                    print(f"Could not find image/label directories for {dataset_name}")
            else:
                print(f"Failed to create dataloaders for {dataset_name}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            gc.collect()
            continue

    print("\nPreprocessing Summary:")
    print("=" * 50)
    print(f"Successfully processed datasets: {len(processed_datasets)}/{len(datasets)}")
    for dataset in processed_datasets:
        print(f"- {dataset}")

    # Save dataloader information
    print("\nSaving dataloader information...")
    save_dataloader_info(dataloaders, save_dir)

    # Load and verify saved dataloaders
    print("\nVerifying saved dataloaders...")
    recreated_loaders = load_and_recreate_dataloaders(save_dir)
    verify_saved_dataloaders(dataloaders, recreated_loaders)

    print("\nPreprocessing and dataloader creation completed!")
    return dataloaders

if __name__ == "__main__":
    dataloaders = main()
