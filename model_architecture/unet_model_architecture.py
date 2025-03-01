import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Avoid operations that could collapse spatial dimensions
        attention = self.conv(x)
        attention = self.activation(attention)
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Add residual connection
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity)  # Residual connection

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        # Load pretrained ResNet34
        resnet = resnet34(pretrained=True)

        # Modify first layer to accept single channel
        self.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize first layer with pretrained weights
        with torch.no_grad():
            self.firstconv.weight[:, 0:1, :, :] = torch.sum(resnet.conv1.weight, dim=1, keepdim=True)

        # Encoder (ResNet layers)
        self.encoder1 = nn.Sequential(
            self.firstconv,
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels

        # Simple attention at deepest level
        self.attention = AttentionBlock(512)

        # Decoder
        self.decoder5 = ConvBlock(512, 512)
        self.decoder4 = ConvBlock(512 + 256, 256)
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 64, 32)

        # Final layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(0.25)

        # Initialize weights of decoder
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.final_conv]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e1_pool = self.pool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Apply simple attention only at the deepest level
        e5 = self.attention(e5)

        # Apply dropout for regularization
        e5 = self.dropout(e5)

        # Decoder with skip connections
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([self.upsample(d5), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))

        # Final output
        final_out = self.final_conv(self.upsample(d1))

        return torch.sigmoid(final_out)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        if self.pos_weight is None:
            # Calculate weights based on inverse class frequency
            neg_count = (target == 0).float().sum()
            pos_count = (target == 1).float().sum()
            total = neg_count + pos_count
            self.pos_weight = (neg_count / total) / (pos_count / total)

        return F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight * torch.ones_like(target)
        )

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.weighted_bce = WeightedBCELoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        weighted_bce = self.weighted_bce(pred, target)

        # Focal Loss component
        pt = torch.exp(-weighted_bce)
        focal = (1 - pt) ** 2 * weighted_bce

        return (self.dice_weight * dice +
                self.bce_weight * weighted_bce
                )

def initialize_model():
    """Initialize model, loss function, and print architecture summary"""
    print("Initializing Model Architecture")
    print("=" * 50)

    try:
        # Initialize model
        model = ResNetUNet(n_classes=1)

        # Create loss function
        criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")

        model = model.to(device)

        # Print model summary
        print("\nModel Architecture:")
        print("-" * 30)

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Test forward pass
        print("\nTesting forward pass...")
        model.eval()  # Set to evaluation mode
        test_input = torch.randn(1, 1, 512, 512).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")

        print("\nModel architecture initialization completed successfully!")
        return model, criterion, device

    except Exception as e:
        print(f"Error in model initialization: {str(e)}")
        return None, None, None

def main():
    print("Starting Model Architecture Setup")
    print("=" * 50)

    try:
        # Initialize model, criterion, and device
        model, criterion, device = initialize_model()

        if model is None:
            print("Model initialization failed!")
            return None, None, None

        # Save model architecture (optional)
        try:
            torch.save(model.state_dict(), '/kaggle/working/initial_model.pth')
            print("\nSaved initial model state")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

        print("\nModel Architecture Setup Completed!")
        return model, criterion, device

    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    model, criterion, device = main()
