import torch
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import distance_transform_edt, binary_erosion
from copy import deepcopy
import gc
import traceback
from vessel_data_preprocessing import (
    VesselDataset, 
    DataLoader,
    get_train_transforms,
    get_val_transforms
)
from unet_model_architecture import (
    ResNetUNet,
    CombinedLoss,
    class_weights
)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_surface_dice(pred, target, tolerance=1):
    """Calculate Surface Dice score with memory-efficient operations"""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    import numpy as np

    # Ensure inputs are boolean
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Get surface points using XOR operation
    pred_eroded = binary_erosion(pred)
    target_eroded = binary_erosion(target)

    pred_surface = np.logical_xor(pred, pred_eroded)
    target_surface = np.logical_xor(target, target_eroded)

    # Calculate distance maps
    pred_distance = distance_transform_edt(~pred_surface)
    target_distance = distance_transform_edt(~target_surface)

    # Get surface points within tolerance
    pred_tolerant = pred_surface & (target_distance <= tolerance)
    target_tolerant = target_surface & (pred_distance <= tolerance)

    # Calculate Surface Dice
    surface_dice = (2.0 * pred_tolerant.sum() + 1e-7) / (pred_surface.sum() + target_surface.sum() + 1e-7)

    return surface_dice.item()

def calculate_metrics(pred, target):
    """Calculate Dice and Surface Dice metrics"""
    pred = pred.float()
    target = target.float()

    # Move tensors to CPU and convert to numpy for Surface Dice
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Regular Dice
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + 1e-7) / (union + 1e-7)

    # Surface Dice (calculate for each sample in batch)
    surface_dice_scores = []
    for p, t in zip(pred_np, target_np):
        surface_dice_scores.append(calculate_surface_dice(p.squeeze(), t.squeeze()))
    avg_surface_dice = np.mean(surface_dice_scores)

    # Clear memory
    del pred_np, target_np
    gc.collect()

    return {
        'dice': dice.item(),
        'surface_dice': avg_surface_dice
    }

def mixup_batch(images, masks, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_masks = lam * masks + (1 - lam) * masks[index]

    return mixed_images, mixed_masks

class Trainer:
    def __init__(self, model, criterion, optimizer, device, save_dir,
                save_every=1, patience=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.patience = patience

        # Initialize best metrics
        self.best_dice = -1
        self.best_surface_dice = -1
        self.patience_counter = 0

        # Initialize history
        self.history = {
            'train_loss': [], 'train_dice': [], 'train_surface_dice': [],
            'val_loss': [], 'val_dice': [], 'val_surface_dice': [],
            'lr': []
        }

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_loader):
        self.model.train()
        losses = AverageMeter()
        dice_scores = AverageMeter()
        surface_dice_scores = AverageMeter()

        pbar = tqdm(train_loader, desc=f'Training')

        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass with mixed precision
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Calculate metrics
                with torch.no_grad():
                    metrics = calculate_metrics((outputs > 0.5).float(), masks)

                # Update meters
                losses.update(loss.item(), images.size(0))
                dice_scores.update(metrics['dice'], images.size(0))
                surface_dice_scores.update(metrics['surface_dice'], images.size(0))

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Dice': f'{dice_scores.avg:.4f}',
                    'SurfDice': f'{surface_dice_scores.avg:.4f}',
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

                # Clear memory
                del images, masks, outputs, loss
                if batch_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue

        return {
            'loss': losses.avg,
            'dice': dice_scores.avg,
            'surface_dice': surface_dice_scores.avg
        }

    def validate(self, val_loader):
        self.model.eval()

        losses = AverageMeter()
        dice_scores = AverageMeter()
        surface_dice_scores = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, masks in pbar:
                try:
                    # Move to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    # Calculate metrics
                    pred_binary = (outputs > 0.5).float()
                    metrics = calculate_metrics(pred_binary, masks)

                    # Update meters
                    losses.update(loss.item(), images.size(0))
                    dice_scores.update(metrics['dice'], images.size(0))
                    surface_dice_scores.update(metrics['surface_dice'], images.size(0))

                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f'{losses.avg:.4f}',
                        'Dice': f'{dice_scores.avg:.4f}',
                        'SurfDice': f'{surface_dice_scores.avg:.4f}'
                    })

                    # Clear memory
                    del images, masks, outputs, pred_binary
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue

        return {
            'loss': losses.avg,
            'dice': dice_scores.avg,
            'surface_dice': surface_dice_scores.avg
        }

    def train(self, train_loader, val_loader, num_epochs, scheduler=None, early_stopping=True):
        """Train the model with safer approach"""
        print(f"Starting safer improved training with {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            print('-' * 20)

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler with validation metrics for ReduceLROnPlateau
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use dice score as the metric to monitor
                scheduler.step(val_metrics['dice'])
            elif scheduler is not None:
                # For other schedulers that don't need metrics
                scheduler.step()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['train_surface_dice'].append(train_metrics['surface_dice'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_surface_dice'].append(val_metrics['surface_dice'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                 f"Dice: {train_metrics['dice']:.4f}, "
                 f"Surface Dice: {train_metrics['surface_dice']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                 f"Dice: {val_metrics['dice']:.4f}, "
                 f"Surface Dice: {val_metrics['surface_dice']:.4f}")

            # Check for improvement
            improved = False
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                improved = True
                print(f"New best Dice: {self.best_dice:.4f}")

            if val_metrics['surface_dice'] > self.best_surface_dice:
                self.best_surface_dice = val_metrics['surface_dice']
                improved = True
                print(f"New best Surface Dice: {self.best_surface_dice:.4f}")

            # Save checkpoint if improved
            if improved:
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print("New best model saved!")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                if early_stopping and self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                    break

            # Plot current progress
            if epoch % 2 == 0 or epoch == num_epochs:
                self.plot_training_history()

            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nTraining completed!")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print(f"Best Surface Dice Score: {self.best_surface_dice:.4f}")

        return self.best_dice, self.best_surface_dice


    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_dice': self.best_dice,
            'best_surface_dice': self.best_surface_dice,
            'history': self.history
        }

        # Save periodic checkpoint
        if epoch % self.save_every == 0:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_model_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model to {best_model_path}")

        # Clear memory
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_dice = checkpoint['best_dice']
            self.best_surface_dice = checkpoint['best_surface_dice']
            self.history = checkpoint['history']
            start_epoch = checkpoint['epoch']

            print(f"Loaded checkpoint from epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return 0

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        print(f"Saved training history to {history_path}")

    def plot_training_history(self):
        """Display training history plots without saving"""
        if len(self.history['train_loss']) < 2:
            print("Not enough data points to plot (need at least 2 epochs)")
            return

        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], 'b-', label='Train', marker='o')
        plt.plot(self.history['val_loss'], 'r-', label='Validation', marker='s')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot metrics
        plt.subplot(2, 1, 2)
        plt.plot(self.history['train_dice'], 'b-', label='Train Dice', marker='o')
        plt.plot(self.history['val_dice'], 'b--', label='Val Dice', marker='s')
        plt.plot(self.history['train_surface_dice'], 'r-', label='Train Surface Dice', marker='^')
        plt.plot(self.history['val_surface_dice'], 'r--', label='Val Surface Dice', marker='v')
        plt.title('Metrics History')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_and_recreate_dataloaders(save_dir):
    """Recreate dataloaders from saved information"""
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

    return dataloaders

def create_subset_loader(loader, subset_size=50):
    """Create a smaller dataloader for testing"""
    subset_dataset = torch.utils.data.Subset(
        loader.dataset,
        indices=range(min(subset_size, len(loader.dataset)))
    )
    return DataLoader(
        subset_dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory
    )

def main(test_mode=False, subset_size=50, num_epochs=15):
    """
    Safer improved training pipeline with minimal changes to your original model

    Args:
        test_mode (bool): If True, use subset of data for testing
        subset_size (int): Number of samples to use in test mode
        num_epochs (int): Number of epochs to train
    """
    print("Starting Safer Improved Training Pipeline")
    print("=" * 50)

    try:
        # Clear initial memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load dataloaders
        print("\nLoading dataloaders...")
        dataloader_path = Path('/kaggle/working/dataloader_info')
        dataloaders = load_and_recreate_dataloaders(dataloader_path)
        print(f"Successfully loaded dataloaders for {len(dataloaders)} datasets")

        if test_mode:
            print(f"\nTest Mode: Creating subset of {subset_size} samples")
            subset_loaders = {}
            for dataset_name, loaders in dataloaders.items():
                subset_loaders[dataset_name] = {
                    'train': create_subset_loader(loaders['train'], subset_size),
                    'val': create_subset_loader(loaders['val'], subset_size//5)
                }
            dataloaders = subset_loaders
            print("Created subset dataloaders for testing")
            gc.collect()

        # Load improved model
        print("\nLoading model with minimal improvements...")
        model = ResNetUNet(n_classes=1)

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        model = model.to(device)

        # Setup save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_prefix = 'test' if test_mode else 'safer_improved'
        base_save_dir = Path(f'/kaggle/working/{mode_prefix}_training_results_{timestamp}')
        base_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated base save directory: {base_save_dir}")

        # Print training configuration
        print("\nTraining Configuration:")
        print(f"Mode: {'Test' if test_mode else 'Safer Improved'}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Device: {device}")
        if test_mode:
            print(f"Subset size: {subset_size}")

        # Initialize results dictionary to store best scores
        results = {}

        # Start training for each dataset
        for dataset_name, loaders in dataloaders.items():
            print(f"\nTraining on dataset: {dataset_name}")
            print(f"Train samples: {len(loaders['train'].dataset)}")
            print(f"Val samples: {len(loaders['val'].dataset)}")
            print("-" * 30)

            # Get dataset-specific weights
            weights = class_weights.get(dataset_name, {'dice_weight': 0.7, 'bce_weight': 0.3})

            criterion = CombinedLoss(
                dice_weight=weights['dice_weight'],
                bce_weight=weights['bce_weight']
            )

            # Setup optimizer - keeping original learning rate
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )

            # Setup learning rate scheduler - simple ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            )

            # Create dataset-specific save directory
            dataset_save_dir = base_save_dir / dataset_name
            dataset_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Save directory for {dataset_name}: {dataset_save_dir}")

            # Create trainer
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                save_dir=dataset_save_dir,
                save_every=1 if test_mode else 3,
                patience=3 if test_mode else 8
            )

            # Train
            best_dice, best_surface_dice = trainer.train(
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                num_epochs=num_epochs,
                scheduler=scheduler,
                early_stopping=True
            )

            # Store results
            results[dataset_name] = {
                'best_dice': best_dice,
                'best_surface_dice': best_surface_dice
            }

            print(f"\nCompleted training for {dataset_name}")
            print(f"Best Dice Score: {best_dice:.4f}")
            print(f"Best Surface Dice Score: {best_surface_dice:.4f}")

            # Save history
            trainer.save_history()

            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear some variables to free memory
            del loaders, trainer
            gc.collect()

        print("\nTraining Pipeline Completed!")
        print("\nFinal Results Summary:")
        print("=" * 50)
        for dataset_name, metrics in results.items():
            print(f"\n{dataset_name}:")
            print(f"Best Dice Score: {metrics['best_dice']:.4f}")
            print(f"Best Surface Dice Score: {metrics['best_surface_dice']:.4f}")

        # Save final results
        results_path = base_save_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved final results to: {results_path}")

        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


if __name__ == "__main__":
    # For testing
    #results = safer_improved_main(test_mode=True, num_epochs=2, subset_size=50)

    # For full training (uncomment to use)
    results = main(test_mode=False, num_epochs=2)
