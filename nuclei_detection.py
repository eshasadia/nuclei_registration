"""
Nuclei detection module implementing various detection methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import cv2
from config import Config


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDetector(nn.Module):
    """U-Net architecture for nuclei detection."""
    
    def __init__(self, in_channels=3, out_channels=1):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Magnification-aware attention module
        self.magnification_attention = MagnificationAttention(64)
        
    def forward(self, x, magnification=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input image tensor
            magnification: Optional magnification level (5x, 10x, 20x, 40x)
            
        Returns:
            Segmentation mask for nuclei
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Apply magnification-aware attention if provided
        if magnification is not None:
            dec1 = self.magnification_attention(dec1, magnification)
        
        # Final segmentation
        out = self.final_conv(dec1)
        return torch.sigmoid(out)


class MagnificationAttention(nn.Module):
    """Attention module that adapts to different magnification levels."""
    
    def __init__(self, channels):
        """
        Initialize the magnification attention module.
        
        Args:
            channels: Number of input channels
        """
        super().__init__()
        
        # One attention branch for each magnification level
        self.mag_5x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        
        self.mag_10x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        
        self.mag_20x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        
        self.mag_40x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        
    def forward(self, x, magnification):
        """
        Apply attention based on magnification level.
        
        Args:
            x: Feature map
            magnification: Magnification level (5, 10, 20, or 40)
            
        Returns:
            Attention-weighted feature map
        """
        # Select the appropriate attention branch
        if magnification == 5:
            attention_map = torch.sigmoid(self.mag_5x(x))
        elif magnification == 10:
            attention_map = torch.sigmoid(self.mag_10x(x))
        elif magnification == 20:
            attention_map = torch.sigmoid(self.mag_20x(x))
        elif magnification == 40:
            attention_map = torch.sigmoid(self.mag_40x(x))
        else:
            # Default to 10x if magnification not specified
            attention_map = torch.sigmoid(self.mag_10x(x))
        
        # Apply attention
        return x * attention_map


def detect_nuclei(image: np.ndarray, model: nn.Module, magnification: int = None) -> Dict:
    """
    Detect nuclei in the input image.
    
    Args:
        image: Input RGB image (numpy array)
        model: Detection model
        magnification: Optional magnification level (5, 10, 20, or 40)
        
    Returns:
        Dictionary with detected nuclei information
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert image to tensor
    if isinstance(image, np.ndarray):
        # Convert numpy array to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = image
    
    # Move to device
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Get segmentation mask
    with torch.no_grad():
        if magnification is not None:
            mask = model(image_tensor, magnification)
        else:
            mask = model(image_tensor)
    
    # Convert mask to numpy array
    mask_np = mask.squeeze().cpu().numpy()
    
    # Threshold the mask
    threshold_mask = (mask_np > Config.INTENSITY_THRESHOLD).astype(np.uint8)
    
    # Find connected components (nuclei)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        threshold_mask, connectivity=8
    )
    
    # Filter out small components
    nuclei = []
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= Config.NUCLEI_MIN_SIZE:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            centroid = centroids[i]
            
            # Get the mask for this nucleus
            nucleus_mask = (labels == i).astype(np.uint8)
            
            # Calculate additional properties
            moments = cv2.moments(nucleus_mask)
            eccentricity = 0  # Default value
            
            # Calculate eccentricity if possible
            if moments['mu20'] != 0 and moments['mu02'] != 0:
                # Calculate eigenvalues of the covariance matrix
                a = moments['mu20'] / moments['m00']
                b = 2 * moments['mu11'] / moments['m00']
                c = moments['mu02'] / moments['m00']
                
                # Calculate eigenvalues
                lambda1 = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + (b / 2) ** 2)
                lambda2 = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + (b / 2) ** 2)
                
                # Calculate eccentricity
                if lambda1 > 0:
                    eccentricity = np.sqrt(1 - (lambda2 / lambda1))
            
            # Append nucleus information
            nuclei.append({
                'id': i,
                'centroid': centroid,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'eccentricity': eccentricity,
                'mask': nucleus_mask
            })
    
    return {
        'original_image': image.astype(np.uint8) if isinstance(image, np.ndarray) else None,
        'detection_mask': threshold_mask,
        'nuclei': nuclei,
        'count': len(nuclei)
    }


class NucleiDetectionPipeline:
    """End-to-end pipeline for detecting nuclei in histology images."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the nuclei detection pipeline.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize the model
        self.model = UNetDetector(in_channels=3, out_channels=1)
        
        # Load weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Move to device
        self.model = self.model.to(self.device)
    
    def detect(self, image: np.ndarray, magnification: int = None) -> Dict:
        """
        Detect nuclei in the input image.
        
        Args:
            image: Input RGB image
            magnification: Optional magnification level (5, 10, 20, or 40)
            
        Returns:
            Dictionary with detected nuclei information
        """
        return detect_nuclei(image, self.model, magnification)
    
    def visualize_detections(self, detection_result: Dict, show_centroids: bool = True) -> np.ndarray:
        """
        Visualize the detected nuclei.
        
        Args:
            detection_result: Output from detect() method
            show_centroids: Whether to show nucleus centroids
            
        Returns:
            Visualization image with detected nuclei
        """
        # Get original image and create a copy for visualization
        original_image = detection_result['original_image']
        if original_image is None:
            raise ValueError("Original image not available in detection result")
        
        vis_image = original_image.copy()
        
        # Create a colored overlay for nuclei
        overlay = np.zeros_like(vis_image)
        
        # Draw each nucleus
        for nucleus in detection_result['nuclei']:
            # Get nucleus mask
            nucleus_mask = nucleus['mask']
            
            # Create colored region for this nucleus
            color = np.random.randint(0, 255, size=3)
            colored_mask = np.zeros_like(vis_image)
            for c in range(3):
                colored_mask[:, :, c] = color[c] * nucleus_mask
            
            # Overlay the nucleus
            overlay = np.maximum(overlay, colored_mask)
            
            # Draw centroid if requested
            if show_centroids:
                centroid = tuple(int(v) for v in nucleus['centroid'])
                cv2.circle(vis_image, centroid, 3, (0, 255, 0), -1)
        
        # Blend original image and overlay
        alpha = 0.4
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
        
        # Draw nucleus count
        cv2.putText(
            vis_image,
            f"Nuclei: {detection_result['count']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return vis_image


if __name__ == "__main__":
    # Test with a sample image
    import os
    import matplotlib.pyplot as plt
    
    # Initialize the pipeline with random weights
    pipeline = NucleiDetectionPipeline()
    
    # Test on a sample image if available
    sample_path = os.path.join(Config.DATA_ROOT, "sample.png")
    if os.path.exists(sample_path):
        # Load image
        image = cv2.imread(sample_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect nuclei
        detection_result = pipeline.detect(image, magnification=10)
        
        # Visualize
        vis_image = pipeline.visualize_detections(detection_result)
        
        # Display results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(vis_image)
        plt.title(f"Detected Nuclei: {detection_result['count']}")
        plt.tight_layout()
        plt.show()
