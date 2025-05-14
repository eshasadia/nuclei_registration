"""
Data preprocessing module for stain normalization and image preparation.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional
from config import Config


class StainNormalizer:
    """Implements multiple stain normalization methods for histopathology images."""
    
    def __init__(self, method: str = "macenko", reference_image_path: Optional[str] = None):
        """
        Initialize the stain normalizer.
        
        Args:
            method: Normalization method (macenko, reinhard, vahadane)
            reference_image_path: Path to the reference image for normalization
        """
        self.method = method.lower()
        self.reference_image = None
        
        if reference_image_path:
            self.set_reference(reference_image_path)
        
        # For storing Macenko method parameters
        self.stain_matrix_ref = None
        self.concentrations_ref = None
        
    def set_reference(self, image_path: str) -> None:
        """Set the reference image for normalization."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        
        self.reference_image = cv2.imread(image_path)
        if self.reference_image is None:
            raise ValueError(f"Could not read reference image: {image_path}")
        
        self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
        
        # Pre-compute reference parameters based on method
        if self.method == "macenko":
            self._compute_macenko_reference()
        elif self.method == "reinhard":
            self._compute_reinhard_reference()
        elif self.method == "vahadane":
            self._compute_vahadane_reference()
            
    def _compute_macenko_reference(self) -> None:
        """Compute Macenko method reference parameters."""
        img = self.reference_image.astype(np.float32) / 255
        
        # Convert to optical density space
        OD = -np.log10(np.maximum(img, 1e-6))
        
        # Remove pixels with OD intensity less than β in any channel
        beta = 0.15
        mask = np.all(OD > beta, axis=2)
        OD = OD[mask]
        
        # Compute eigenvectors
        _, V = np.linalg.eigh(np.cov(OD.T))
        V = V[:, [1, 2]]  # Eigenvectors corresponding to the two largest eigenvalues
        
        # Project on the eigenvectors
        That = np.dot(OD, V)
        
        # Angular coordinates in the projected 2D subspace
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find the min and max angles
        minPhi = np.percentile(phi, 1)
        maxPhi = np.percentile(phi, 99)
        
        # Find the eigenvectors that correspond to the min and max angles
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # Build the stain matrix
        self.stain_matrix_ref = np.vstack([v1, v2, np.cross(v1, v2)]).T
        
        # Compute concentrations
        OD_flat = OD.reshape((-1, 3))
        self.concentrations_ref = np.linalg.lstsq(self.stain_matrix_ref, OD_flat.T, rcond=None)[0].T
        
    def _compute_reinhard_reference(self) -> None:
        """Compute Reinhard method reference parameters."""
        # Convert to LAB color space
        lab = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2LAB)
        
        # Compute mean and std for each channel
        self.lab_mean = np.mean(lab, axis=(0, 1))
        self.lab_std = np.std(lab, axis=(0, 1))
        
    def _compute_vahadane_reference(self) -> None:
        """Compute Vahadane method reference parameters."""
        # Implementation of Vahadane method parameters
        # This is a simplified version - for a complete implementation, 
        # use or adapt existing libraries like histomicstk
        pass
            
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an input image using the specified method.
        
        Args:
            image: Input RGB image
            
        Returns:
            Normalized RGB image
        """
        if self.reference_image is None:
            raise ValueError("Reference image not set. Call set_reference() first.")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3:
            raise ValueError(f"Unexpected image format with {image.shape[2]} channels")
        
        # Apply the selected normalization method
        if self.method == "macenko":
            return self._normalize_macenko(image)
        elif self.method == "reinhard":
            return self._normalize_reinhard(image)
        elif self.method == "vahadane":
            return self._normalize_vahadane(image)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _normalize_macenko(self, image: np.ndarray) -> np.ndarray:
        """Normalize using Macenko method."""
        img = image.astype(np.float32) / 255
        
        # Convert to optical density space
        OD = -np.log10(np.maximum(img, 1e-6))
        
        # Reshape to one row per pixel
        h, w, _ = OD.shape
        OD = OD.reshape((-1, 3))
        
        # Determine stain vectors
        _, V = np.linalg.eigh(np.cov(OD.T))
        V = V[:, [1, 2]]  # Eigenvectors for the two largest eigenvalues
        
        # Project on the eigenvectors
        That = np.dot(OD, V)
        
        # Angular coordinates
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find the min and max angles
        minPhi = np.percentile(phi, 1)
        maxPhi = np.percentile(phi, 99)
        
        # Eigenvectors for min and max angles
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # Build the stain matrix
        stain_matrix_src = np.vstack([v1, v2, np.cross(v1, v2)]).T
        
        # Get source image concentrations
        C_src = np.linalg.lstsq(stain_matrix_src, OD.T, rcond=None)[0].T
        
        # Scale to match target concentrations
        C_src[:, 0] = C_src[:, 0] * np.median(self.concentrations_ref[:, 0]) / np.median(C_src[:, 0])
        C_src[:, 1] = C_src[:, 1] * np.median(self.concentrations_ref[:, 1]) / np.median(C_src[:, 1])
        
        # Recreate the image using the reference stain matrix
        OD_norm = np.dot(C_src, self.stain_matrix_ref.T)
        
        # Convert back to RGB
        img_norm = 10 ** (-OD_norm)
        img_norm = np.clip(img_norm, 0, 1)
        img_norm = (img_norm * 255).astype(np.uint8)
        
        # Reshape back
        img_norm = img_norm.reshape((h, w, 3))
        
        return img_norm
        
    def _normalize_reinhard(self, image: np.ndarray) -> np.ndarray:
        """Normalize using Reinhard method."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute mean and std for each channel
        lab_mean = np.mean(lab, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        
        # Scale to match target
        lab = ((lab - lab_mean) * (self.lab_std / lab_std)) + self.lab_mean
        
        # Convert back to RGB
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Clip values to valid range
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized
        
    def _normalize_vahadane(self, image: np.ndarray) -> np.ndarray:
        """Normalize using Vahadane method."""
        # Implementation of Vahadane normalization
        # Simplified version - for a complete implementation, 
        # use or adapt existing libraries like histomicstk
        return image  # Placeholder


class HistologyDataset(Dataset):
    """Dataset for histology images with preprocessing capabilities."""
    
    def __init__(
        self, 
        data_dir: str,
        is_training: bool = True,
        transform = None,
        normalizer: Optional[StainNormalizer] = None,
        load_pairs: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the histology images
            is_training: Whether this is for training (affects augmentation)
            transform: Optional transforms to apply
            normalizer: Optional stain normalizer to apply
            load_pairs: Whether to load image pairs (for registration)
        """
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = transform
        self.normalizer = normalizer
        self.load_pairs = load_pairs
        
        # Find all images
        self.image_paths = []
        for ext in ["*.png", "*.jpg", "*.tif", "*.tiff"]:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        
        # Sort for reproducibility
        self.image_paths.sort()
        
        if self.load_pairs:
            # Creating pairs for registration training
            # For simplicity, we'll create pairs from adjacent images
            # In a real implementation, this would depend on your dataset organization
            self.pairs = []
            for i in range(0, len(self.image_paths) - 1, 2):
                if i + 1 < len(self.image_paths):
                    self.pairs.append((self.image_paths[i], self.image_paths[i + 1]))
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.load_pairs:
            return len(self.pairs)
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the images and metadata
        """
        if self.load_pairs:
            # Load a pair of images for registration
            fixed_path, moving_path = self.pairs[idx]
            fixed_img = self._load_and_preprocess(fixed_path)
            moving_img = self._load_and_preprocess(moving_path)
            
            # Metadata
            metadata = {
                'fixed_path': fixed_path,
                'moving_path': moving_path,
                'fixed_shape': fixed_img.shape,
                'moving_shape': moving_img.shape
            }
            
            # Convert to tensor
            if self.transform:
                fixed_img = self.transform(fixed_img)
                moving_img = self.transform(moving_img)
            
            return {
                'fixed': fixed_img,
                'moving': moving_img,
                'metadata': metadata
            }
        else:
            # Load a single image
            img_path = self.image_paths[idx]
            img = self._load_and_preprocess(img_path)
            
            # Metadata
            metadata = {
                'path': img_path,
                'shape': img.shape
            }
            
            # Convert to tensor
            if self.transform:
                img = self.transform(img)
            
            return {
                'image': img,
                'metadata': metadata
            }
    
    def _load_and_preprocess(self, img_path: str) -> np.ndarray:
        """Load and preprocess an image."""
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
        
        # Apply stain normalization if available
        if self.normalizer:
            img = self.normalizer.normalize(img)
        
        return img


def create_dataloader(
    data_dir: str, 
    batch_size: int, 
    is_training: bool = True,
    normalizer: Optional[StainNormalizer] = None
) -> DataLoader:
    """
    Create a DataLoader for the histology dataset.
    
    Args:
        data_dir: Directory containing the images
        batch_size: Batch size
        is_training: Whether this is for training
        normalizer: Optional stain normalizer
        
    Returns:
        DataLoader for the dataset
    """
    # Define transforms
    if is_training:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(Config.TARGET_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(Config.TARGET_SIZE),
            transforms.ToTensor(),
        ])
    
    # Create dataset
    dataset = HistologyDataset(
        data_dir=data_dir,
        is_training=is_training,
        transform=transform,
        normalizer=normalizer,
        load_pairs=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the stain normalizer
    normalizer = StainNormalizer(method=Config.STAIN_NORM_METHOD)
    
    if os.path.exists(Config.REFERENCE_STAIN_PATH):
        normalizer.set_reference(Config.REFERENCE_STAIN_PATH)
        
        # Test on a sample image
        sample_path = os.path.join(Config.DATA_ROOT, "sample.png")
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            normalized = normalizer.normalize(img)
            
            # Display results
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(normalized)
            plt.title(f"Normalized ({Config.STAIN_NORM_METHOD})")
            plt.tight_layout()
            plt.show()
