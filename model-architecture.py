"""
Source Code of:
"Hierarchical Attention Lightweight U-Net for Gastro-Intestinal Tract Segmentation"
Author: Marreddi Jayanth Sai and Narinder Singh Punn
Date: Feb 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution as described in the research paper"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSC(nn.Module):
    """Double Depthwise Separable Convolution Block (DDSC)"""
    def __init__(self, in_channels, out_channels):
        super(DoubleDSC, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x

class ChannelSpatialAttention(nn.Module):
    """Hierarchical Channel-Spatial Attention (CSA) mechanism"""
    def __init__(self, channels):
        super(ChannelSpatialAttention, self).__init__()
        
        # Channel attention - Bottleneck Convolution Block (BCB)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.channel_conv2 = nn.Conv2d(channels // 4, channels, 1, bias=False)
        self.channel_bn = nn.BatchNorm2d(channels)
        
        # Spatial attention - Dilated Convolution Block (DCB)  
        self.spatial_conv1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.spatial_dsc1 = DepthwiseSeparableConv(channels // 4, channels // 4, 
                                                  kernel_size=3, padding=4, dilation=4)
        self.spatial_dsc2 = DepthwiseSeparableConv(channels // 4, channels // 4, 
                                                  kernel_size=3, padding=4, dilation=4)
        self.spatial_conv2 = nn.Conv2d(channels // 4, 1, 1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        # Store original input for residual connection
        original_x = x
        
        # Channel attention pathway
        channel_att = self.global_avg_pool(x)  # Global average pooling
        channel_att = self.channel_conv1(channel_att)  # Bottleneck: C -> C/4
        channel_att = self.channel_conv2(channel_att)  # Expand: C/4 -> C
        channel_att = self.channel_bn(channel_att)
        channel_att = F.relu(channel_att)  # A_c
        
        # Spatial attention pathway
        spatial_att = self.spatial_conv1(x)  # 1x1 conv: C -> C/4
        spatial_att = self.spatial_dsc1(spatial_att)  # 3x3 dilated DSC
        spatial_att = self.spatial_dsc2(spatial_att)  # 3x3 dilated DSC  
        spatial_att = self.spatial_conv2(spatial_att)  # 1x1 conv: C/4 -> 1
        spatial_att = self.spatial_bn(spatial_att)
        spatial_att = F.relu(spatial_att)  # A_s
        
        # Combine attentions: F_a = F_i ⊙ (A_c + A_s) + F_i
        attention = channel_att + spatial_att
        out = original_x * attention + original_x
        
        return out

class EncoderBlock(nn.Module):
    """Encoder block with DDSC and MaxPooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.ddsc = DoubleDSC(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        features = self.ddsc(x)
        pooled = self.pool(features)
        return features, pooled

class DecoderBlock(nn.Module):
    """Decoder block with upsampling, CSA attention, and DDSC"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.csa = ChannelSpatialAttention(in_channels // 2 + skip_channels)
        self.ddsc = DoubleDSC(in_channels // 2 + skip_channels, out_channels)
        
    def forward(self, x, skip):
        # Upsample using transposed convolution
        x = self.upconv(x)
        
        # Ensure spatial dimensions match for concatenation
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply hierarchical channel-spatial attention
        x = self.csa(x)
        
        # Apply double DSC for feature refinement
        x = self.ddsc(x)
        
        return x

class HALUNet(nn.Module):
    """
    Hierarchical Attention Lightweight U-Net (HALU-Net) for GI Tract Segmentation
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale medical images)
        out_channels: Number of output classes (default: 3 for stomach, small bowel, large bowel)
    """
    def __init__(self, in_channels=1, out_channels=3):
        super(HALUNet, self).__init__()
        
        # Encoder path with reduced channels for efficiency
        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64) 
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = DoubleDSC(256, 512)
        
        # Decoder path with hierarchical attention
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)
        
        # Final output layer
        self.final_conv = DepthwiseSeparableConv(32, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # Encoder path with skip connections
        skip1, x = self.enc1(x)    # 256x256 -> 128x128, skip: 32 channels
        skip2, x = self.enc2(x)    # 128x128 -> 64x64, skip: 64 channels
        skip3, x = self.enc3(x)    # 64x64 -> 32x32, skip: 128 channels  
        skip4, x = self.enc4(x)    # 32x32 -> 16x16, skip: 256 channels
        
        # Bottleneck
        x = self.bottleneck(x)     # 16x16, 512 channels
        
        # Decoder path with hierarchical attention at each stage
        x = self.dec4(x, skip4)    # 16x16 -> 32x32, 256 channels
        x = self.dec3(x, skip3)    # 32x32 -> 64x64, 128 channels
        x = self.dec2(x, skip2)    # 64x64 -> 128x128, 64 channels
        x = self.dec1(x, skip1)    # 128x128 -> 256x256, 32 channels
        
        # Final segmentation output
        x = self.final_conv(x)     # 256x256, out_channels
        
        return x

class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss as used in the research paper"""
    def __init__(self, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.gamma = gamma
        
    def focal_loss(self, pred, target):
        """Focal Loss to handle class imbalance"""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred, target):
        """Dice Loss for segmentation"""
        pred = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
        
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return focal + dice

def create_model(in_channels=1, out_channels=3):
    """
    Factory function to create HALU-Net model
    
    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output classes (default: 3)
        
    Returns:
        HALUNet model instance
    """
    return HALUNet(in_channels=in_channels, out_channels=out_channels)

def get_loss_function(gamma=2.0):
    """
    Get the combined loss function used in the research paper
    
    Args:
        gamma: Focal loss gamma parameter (default: 2.0)
        
    Returns:
        CombinedLoss instance
    """
    return CombinedLoss(gamma=gamma)