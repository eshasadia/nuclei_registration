"""
Main model class that integrates all components of the nuclei registration system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import cv2
import os

from config import Config
from data_preprocessing import StainNormalizer
from nuclei_detection import UNetDetector, detect_nuclei
from feature_extraction import extract_features
from registration import RegistrationModule, visualize_registration


class NucleiRegistrationModel(nn.Module):
    """
    End-to-end model for nuclei-based registration of histology images.
    """
    
    def __init__(
        self,
        detector_weights_path: Optional[str] = None,
        stain_norm_method: str = None,
        reference_stain_path: str = None,
        registration_method: str = None,
        device: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            detector_weights_path: Path to pre-trained detector weights
            stain_norm_method: Stain normalization method
            reference_stain_path: Path to reference stain image
            registration_method: Registration method
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize stain normalizer
        self.stain_norm_method = stain_norm_method if stain_norm_method else Config.STAIN_NORM_METHOD
        self.normalizer = StainNormalizer(method=self.stain_norm_method)
        
        if reference_stain_path and os.path.exists(reference_stain_path):
            self.normalizer.set_reference(reference_stain_path)
        elif Config.REFERENCE_STAIN_PATH and os.path.exists(Config.REFERENCE_STAIN_PATH):
            self.normalizer.set_reference(Config.REFERENCE_STAIN_PATH)
        
        # Initialize nuclei detector
        self.detector = UNetDetector(in_channels=3, out_channels=1)
        
        # Load detector weights if provided
        if detector_weights_path and os.path.exists(detector_weights_path):
            self.detector.load_state_dict(
                torch.load(detector_weights_path, map_location=self.device)
            )
        
        # Move detector to device
        self.detector = self.detector.to(self.device)
        
        # Initialize registration module
        self.registration_method = registration_method if registration_method else Config.REGISTRATION_METHOD
        self.registration = RegistrationModule(method=self.registration_method)
        
        # Set to evaluation mode by default
        self.eval()
    
    def forward(
        self,
        fixed_image: torch.Tensor,
        moving_image: torch.Tensor,
        fixed_magnification: Optional[int] = None,
        moving_magnification: Optional[int] = None
    ) -> Dict:
        """
        Perform registration of moving image to fixed image.
        
        Args:
            fixed_image: Fixed (target) image
            moving_image: Moving (source) image
            fixed_magnification: Magnification level of fixed image
            moving_magnification: Magnification level of moving image
            
        Returns:
            Registration results
        """
        # Convert to numpy if tensors
        if isinstance(fixed_image, torch.Tensor):
            fixed_image_np = fixed_image.permute(0, 2, 3, 1)[0].cpu().numpy()
        else:
            fixed_image_np = fixed_image
        
        if isinstance(moving_image, torch.Tensor):
            moving_image_np = moving_image.permute(0, 2, 3, 1)[0].cpu().numpy()
        else:
            moving_image_np = moving_image
        
        # Apply stain normalization if reference is set
        if hasattr(self.normalizer, 'reference_image') and self.normalizer.reference_image is not None:
            fixed_image_norm = self.normalizer.normalize(fixed_image_np)
            moving_image_norm = self.normalizer.normalize(moving_image_np)
        else:
            fixed_image_norm = fixed_image_np
            moving_image_norm = moving_image_np
        
        # Detect nuclei in both images
        with torch.no_grad():
            fixed_nuclei = detect_nuclei(
                fixed_image_norm,
                self.detector,
                magnification=fixed_magnification
            )
            
            moving_nuclei = detect_nuclei(
                moving_image_norm,
                self.detector,
                magnification=moving_magnification
            )
        
        # Extract features from detected nuclei
        fixed_features = extract_features(fixed_nuclei['nuclei'], fixed_image_norm)
        moving_features = extract_features(moving_nuclei['nuclei'], moving_image_norm)
        
        # Register nuclei centroids
        registration_result = self.registration.register(
            moving_features['centroids'],
            fixed_features['centroids'],
            moving_features['features']['all'],
            fixed_features['features']['all']
        )
        
        # Transform moving image using the registration result
        if registration_result['transformation'] is not None:
            transformed_moving = self.registration.transform_image(
                moving_image_np,
                registration_result['transformation']
            )
        else:
            transformed_moving = moving_image_np
        
        # Create visualizations
        visualizations = visualize_registration(
            fixed_image_np,
            moving_image_np,
            registration_result
        )
        
        # Return results
        return {
            'fixed_image': fixed_image_np,
            'moving_image': moving_image_np,
            'fixed_nuclei': fixed_nuclei,
            'moving_nuclei': moving_nuclei,
            'fixed_features': fixed_features,
            'moving_features': moving_features,
            'registration_result': registration_result,
            'transformed_moving': transformed_moving,
            'visualizations': visualizations
        }
    
    def register_images(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_magnification: Optional[int] = None,
        moving_magnification: Optional[int] = None
    ) -> Dict:
        """
        Convenient method to register two images.
        
        Args:
            fixed_image: Fixed (target) image
            moving_image: Moving (source) image
            fixed_magnification: Magnification level of fixed image
            moving_magnification: Magnification level of moving image
            
        Returns:
            Registration results
        """
        return self.forward(
            fixed_image,
            moving_image,
            fixed_magnification,
            moving_magnification
        )


class EndToEndTrainableModel(nn.Module):
    """
    End-to-end trainable model for nuclei-based registration.
    """
    
    def __init__(
        self,
        detector_weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the trainable model.
        
        Args:
            detector_weights_path: Path to pre-trained detector weights
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize detector
        self.detector = UNetDetector(in_channels=3, out_channels=1)
        
        # Load detector weights if provided
        if detector_weights_path and os.path.exists(detector_weights_path):
            self.detector.load_state_dict(
                torch.load(detector_weights_path, map_location=self.device)
            )
        
        # Move detector to device
        self.detector = self.detector.to(self.device)
        
        # Learnable feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Trainable registration module
        self.registration_network = RegistrationNetwork()
    
    def forward(
        self,
        fixed_image: torch.Tensor,
        moving_image: torch.Tensor,
        fixed_magnification: Optional[int] = None,
        moving_magnification: Optional[int] = None
    ) -> Dict:
        """
        Perform registration in a differentiable manner.
        
        Args:
            fixed_image: Fixed (target) image
            moving_image: Moving (source) image
            fixed_magnification: Magnification level of fixed image
            moving_magnification: Magnification level of moving image
            
        Returns:
            Registration results including transformation parameters
        """
        # Ensure input is tensor
        if not isinstance(fixed_image, torch.Tensor):
            fixed_image = torch.from_numpy(fixed_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        if not isinstance(moving_image, torch.Tensor):
            moving_image = torch.from_numpy(moving_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Detect nuclei
        with torch.no_grad():
            fixed_mask = self.detector(fixed_image, fixed_magnification)
            moving_mask = self.detector(moving_image, moving_magnification)
        
        # Extract features
        fixed_features = self.feature_extractor(fixed_image, fixed_mask)
        moving_features = self.feature_extractor(moving_image, moving_mask)
        
        # Predict transformation
        transformation_params = self.registration_network(fixed_features, moving_features)
        
        # Apply transformation (differentiable)
        transformed_moving = self.apply_transformation(moving_image, transformation_params)
        
        return {
            'fixed_image': fixed_image,
            'moving_image': moving_image,
            'transformed_moving': transformed_moving,
            'transformation_params': transformation_params,
            'fixed_features': fixed_features,
            'moving_features': moving_features
        }
    
    def apply_transformation(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to image in a differentiable manner.
        
        Args:
            image: Input image tensor
            params: Transformation parameters
            
        Returns:
            Transformed image tensor
        """
        # Extract transformation parameters
        # For rigid transformation, params contains [theta, tx, ty, s]
        theta = params[0]  # Rotation angle
        tx = params[1]     # Translation x
        ty = params[2]     # Translation y
        s = params[3]      # Scale
        
        # Create transformation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Scale * Rotation
        a = s * cos_theta
        b = s * sin_theta
        
        # Create affine matrix
        affine_matrix = torch.zeros(3, 3, device=self.device)
        affine_matrix[0, 0] = a
        affine_matrix[0, 1] = b
        affine_matrix[0, 2] = tx
        affine_matrix[1, 0] = -b
        affine_matrix[1, 1] = a
        affine_matrix[1, 2] = ty
        affine_matrix[2, 2] = 1.0
        
        # Convert to 2x3 matrix for grid_sample
        affine_matrix = affine_matrix[:2, :].unsqueeze(0)
        
        # Get image dimensions
        batch_size, channels, height, width = image.shape
        
        # Create normalized mesh grid
        grid = F.affine_grid(
            affine_matrix,
            (batch_size, channels, height, width),
            align_corners=True
        )
        
        # Apply transformation using grid sampling
        transformed = F.grid_sample(
            image,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return transformed


class FeatureExtractor(nn.Module):
    """
    Learnable feature extractor for nuclei images.
    """
    
    def __init__(self, in_channels=3, out_channels=64):
        """
        Initialize the feature extractor.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output feature channels
        """
        super().__init__()
        
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            image: Input image tensor
            mask: Nuclei segmentation mask
            
        Returns:
            Feature tensor
        """
        # Concatenate image and mask
        x = torch.cat([image, mask], dim=1)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        return x


class RegistrationNetwork(nn.Module):
    """
    Network to predict transformation parameters from features.
    """
    
    def __init__(self, in_channels=64):
        """
        Initialize the registration network.
        
        Args:
            in_channels: Number of input feature channels
        """
        super().__init__()
        
        # Feature processing
        self.conv1 = nn.Conv2d(in_channels * 2, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # [theta, tx, ty, s]
    
    def forward(self, fixed_features: torch.Tensor, moving_features: torch.Tensor) -> torch.Tensor:
        """
        Predict transformation parameters.
        
        Args:
            fixed_features: Features from fixed image
            moving_features: Features from moving image
            
        Returns:
            Transformation parameters [theta, tx, ty, s]
        """
        # Feature concatenation
        x = torch.cat([fixed_features, moving_features], dim=1)
        
        # Feature processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Scale outputs appropriately
        # theta: rotation angle (-pi to pi)
        # tx, ty: translation (-0.5 to 0.5 relative to image size)
        # s: scale (0.5 to 2.0)
        theta = torch.tanh(x[:, 0]) * np.pi
        tx = torch.tanh(x[:, 1]) * 0.5
        ty = torch.tanh(x[:, 2]) * 0.5
        s = torch.sigmoid(x[:, 3]) * 1.5 + 0.5
        
        return torch.stack([theta, tx, ty, s], dim=1)


if __name__ == "__main__":
    # Test the model with sample images
    import os
    import matplotlib.pyplot as plt
    
    # Initialize the model
    model = NucleiRegistrationModel()
    
    # Load sample images
    sample_dir = Config.DATA_ROOT
    
    sample1_path = os.path.join(sample_dir, "sample1.png")
    sample2_path = os.path.join(sample_dir, "sample2.png")
    
    if os.path.exists(sample1_path) and os.path.exists(sample2_path):
        # Load images
        image1 = cv2.imread(sample1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        image2 = cv2.imread(sample2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        # Register images
        result = model.register_images(image2, image1, fixed_magnification=20, moving_magnification=20)
        
        # Display results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(result['moving_image'])
        plt.title("Moving Image")
        
        plt.subplot(2, 3, 2)
        plt.imshow(result['fixed_image'])
        plt.title("Fixed Image")
        
        plt.subplot(2, 3, 3)
        plt.imshow(result['transformed_moving'])
        plt.title("Transformed Moving")
        
        plt.subplot(2, 3, 4)
        plt.imshow(result['visualizations']['moving_points'])
        plt.title("Moving Points")
        
        plt.subplot(2, 3, 5)
        plt.imshow(result['visualizations']['fixed_points'])
        plt.title("Fixed Points")
        
        plt.subplot(2, 3, 6)
        if 'overlay' in result['visualizations']:
            plt.imshow(result['visualizations']['overlay'])
            plt.title("Overlay")
        else:
            plt.imshow(result['visualizations']['transformed_points'])
            plt.title("Transformed Points")
        
        plt.tight_layout()
        plt.show()
