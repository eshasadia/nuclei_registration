"""
Feature extraction from detected nuclei for registration purposes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import cv2
from scipy.spatial import Delaunay
from config import Config


def extract_geometric_features(nucleus: Dict) -> np.ndarray:
    """
    Extract geometric features from a nucleus.
    
    Args:
        nucleus: Dictionary containing nucleus information
        
    Returns:
        Array of geometric features
    """
    # Extract basic geometric features
    area = nucleus['area']
    width = nucleus['width']
    height = nucleus['height']
    eccentricity = nucleus['eccentricity']
    
    # Calculate aspect ratio
    aspect_ratio = width / max(height, 1)  # Avoid division by zero
    
    # Calculate compactness (circularity)
    perimeter = cv2.arcLength(
        cv2.findContours(nucleus['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0],
        True
    )
    compactness = (4 * np.pi * area) / max(perimeter * perimeter, 1e-6)
    
    # Calculate convexity
    hull = cv2.convexHull(
        cv2.findContours(nucleus['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    )
    hull_area = cv2.contourArea(hull)
    convexity = area / max(hull_area, 1)  # Avoid division by zero
    
    # Calculate solidity
    solidity = area / max(hull_area, 1)  # Avoid division by zero
    
    # Calculate equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    # Combine all geometric features
    features = np.array([
        area,
        width,
        height,
        eccentricity,
        aspect_ratio,
        compactness,
        convexity,
        solidity,
        equivalent_diameter
    ])
    
    return features


def extract_intensity_features(nucleus: Dict, image: np.ndarray) -> np.ndarray:
    """
    Extract intensity features from a nucleus.
    
    Args:
        nucleus: Dictionary containing nucleus information
        image: Original image (RGB or grayscale)
        
    Returns:
        Array of intensity features
    """
    # Get nucleus mask and region
    mask = nucleus['mask']
    x, y = nucleus['x'], nucleus['y']
    w, h = nucleus['width'], nucleus['height']
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    x_end = min(image.shape[1], x + w)
    y_end = min(image.shape[0], y + h)
    
    # Extract region of interest
    roi = image[y:y_end, x:x_end]
    
    # If the mask dimensions don't match the ROI, resize the mask
    if mask.shape != (y_end - y, x_end - x):
        mask_roi = mask[y:y_end, x:x_end]
    else:
        mask_roi = mask
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray_roi = roi
    
    # Apply mask
    masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
    
    # Calculate intensity statistics
    non_zero = masked_roi[masked_roi > 0]
    if len(non_zero) == 0:
        return np.zeros(5)  # Return zeros if no valid pixels
    
    mean_intensity = np.mean(non_zero)
    std_intensity = np.std(non_zero)
    min_intensity = np.min(non_zero)
    max_intensity = np.max(non_zero)
    median_intensity = np.median(non_zero)
    
    # Combine intensity features
    features = np.array([
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        median_intensity
    ])
    
    return features


def extract_texture_features(nucleus: Dict, image: np.ndarray) -> np.ndarray:
    """
    Extract texture features from a nucleus.
    
    Args:
        nucleus: Dictionary containing nucleus information
        image: Original image (RGB or grayscale)
        
    Returns:
        Array of texture features
    """
    # Get nucleus mask and region
    mask = nucleus['mask']
    x, y = nucleus['x'], nucleus['y']
    w, h = nucleus['width'], nucleus['height']
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    x_end = min(image.shape[1], x + w)
    y_end = min(image.shape[0], y + h)
    
    # Extract region of interest
    roi = image[y:y_end, x:x_end]
    
    # If the mask dimensions don't match the ROI, resize the mask
    if mask.shape != (y_end - y, x_end - x):
        mask_roi = mask[y:y_end, x:x_end]
    else:
        mask_roi = mask
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray_roi = roi
    
    # Apply mask
    masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
    
    # Check if there are enough pixels to calculate texture features
    if np.sum(mask_roi) < 25:  # Minimum size for GLCM
        return np.zeros(6)  # Return zeros if ROI is too small
    
    # Calculate GLCM-based texture features
    # Using a simplified approach here for efficiency
    glcm = calculate_glcm(masked_roi, mask_roi)
    
    # Calculate texture metrics from GLCM
    contrast = calculate_contrast(glcm)
    dissimilarity = calculate_dissimilarity(glcm)
    homogeneity = calculate_homogeneity(glcm)
    energy = calculate_energy(glcm)
    correlation = calculate_correlation(glcm)
    entropy = calculate_entropy(glcm)
    
    # Combine texture features
    features = np.array([
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        entropy
    ])
    
    return features


def calculate_glcm(image: np.ndarray, mask: np.ndarray = None, distances: List[int] = [1], angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
    """
    Calculate the Gray Level Co-occurrence Matrix (GLCM) for an image.
    
    Args:
        image: Grayscale image
        mask: Optional binary mask
        distances: List of distances to consider
        angles: List of angles to consider
        
    Returns:
        GLCM matrix
    """
    # Quantize image to 8 gray levels to reduce computation
    levels = 8
    max_val = image.max()
    if max_val == 0:
        return np.zeros((levels, levels))
    
    # Quantize the image
    bins = np.linspace(0, max_val, levels + 1)
    quantized = np.digitize(image, bins) - 1
    quantized[quantized < 0] = 0
    
    # Initialize GLCM
    glcm = np.zeros((levels, levels))
    
    # Apply mask if provided
    if mask is not None:
        mask_indices = np.where(mask > 0)
        coords = list(zip(mask_indices[0], mask_indices[1]))
    else:
        h, w = image.shape
        coords = [(i, j) for i in range(h) for j in range(w)]
    
    # Calculate GLCM for each distance and angle
    for distance in distances:
        for angle in angles:
            dx = int(np.round(distance * np.cos(angle)))
            dy = int(np.round(distance * np.sin(angle)))
            
            # Count co-occurrences
            for i, j in coords:
                ni, nj = i + dy, j + dx
                if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                    if mask is None or (mask[i, j] > 0 and mask[ni, nj] > 0):
                        glcm[quantized[i, j], quantized[ni, nj]] += 1
    
    # Normalize GLCM
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()
    
    return glcm


def calculate_contrast(glcm: np.ndarray) -> float:
    """Calculate contrast from GLCM."""
    n = glcm.shape[0]
    contrast = 0
    for i in range(n):
        for j in range(n):
            contrast += glcm[i, j] * ((i - j) ** 2)
    return contrast


def calculate_dissimilarity(glcm: np.ndarray) -> float:
    """Calculate dissimilarity from GLCM."""
    n = glcm.shape[0]
    dissimilarity = 0
    for i in range(n):
        for j in range(n):
            dissimilarity += glcm[i, j] * abs(i - j)
    return dissimilarity


def calculate_homogeneity(glcm: np.ndarray) -> float:
    """Calculate homogeneity from GLCM."""
    n = glcm.shape[0]
    homogeneity = 0
    for i in range(n):
        for j in range(n):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
    return homogeneity


def calculate_energy(glcm: np.ndarray) -> float:
    """Calculate energy from GLCM."""
    return np.sum(glcm ** 2)


def calculate_correlation(glcm: np.ndarray) -> float:
    """Calculate correlation from GLCM."""
    n = glcm.shape[0]
    
    # Calculate means and standard deviations
    i_indices = np.arange(n).reshape(-1, 1)
    j_indices = np.arange(n).reshape(1, -1)
    
    mean_i = np.sum(i_indices * glcm)
    mean_j = np.sum(j_indices * glcm)
    
    std_i = np.sqrt(np.sum(glcm * ((i_indices - mean_i) ** 2)))
    std_j = np.sqrt(np.sum(glcm * ((j_indices - mean_j) ** 2)))
    
    if std_i == 0 or std_j == 0:
        return 0
    
    # Calculate correlation
    correlation = 0
    for i in range(n):
        for j in range(n):
            correlation += glcm[i, j] * ((i - mean_i) * (j - mean_j)) / (std_i * std_j)
    
    return correlation


def calculate_entropy(glcm: np.ndarray) -> float:
    """Calculate entropy from GLCM."""
    # Replace zero values with a small number to avoid log(0)
    epsilon = 1e-10
    glcm_log = np.log2(glcm + epsilon)
    entropy = -np.sum(glcm * glcm_log)
    return entropy


def extract_neighborhood_features(nuclei: List[Dict]) -> np.ndarray:
    """
    Extract features based on the neighborhood of nuclei.
    
    Args:
        nuclei: List of dictionaries containing nucleus information
        
    Returns:
        Array of neighborhood features for each nucleus
    """
    if len(nuclei) < 3:
        # Not enough nuclei for triangulation
        return np.zeros((len(nuclei), 3))
    
    # Get centroids of all nuclei
    centroids = np.array([nucleus['centroid'] for nucleus in nuclei])
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(centroids)
    except:
        # Error in triangulation
        return np.zeros((len(nuclei), 3))
    
    # Initialize neighborhood features
    neighborhood_features = np.zeros((len(nuclei), 3))
    
    # Count neighbors for each nucleus
    neighbor_count = np.zeros(len(nuclei), dtype=int)
    
    # Calculate average distance to neighbors
    avg_distance = np.zeros(len(nuclei))
    
    # Calculate local density
    density = np.zeros(len(nuclei))
    
    # Process each simplex (triangle) in the triangulation
    for simplex in tri.simplices:
        # For each vertex in the simplex
        for i in range(3):
            # Get the other two vertices (neighbors)
            idx = simplex[i]
            neighbor1 = simplex[(i + 1) % 3]
            neighbor2 = simplex[(i + 2) % 3]
            
            # Update neighbor count
            neighbor_count[idx] += 2
            
            # Calculate distances to neighbors
            dist1 = np.linalg.norm(centroids[idx] - centroids[neighbor1])
            dist2 = np.linalg.norm(centroids[idx] - centroids[neighbor2])
            
            # Update total distance
            avg_distance[idx] += (dist1 + dist2)
    
    # Calculate average distance to neighbors
    for i in range(len(nuclei)):
        if neighbor_count[i] > 0:
            avg_distance[i] /= neighbor_count[i]
    
    # Calculate local density (number of nuclei per unit area)
    for i in range(len(nuclei)):
        if avg_distance[i] > 0:
            # Approximate local area as a circle with radius = average distance
            area = np.pi * (avg_distance[i] ** 2)
            density[i] = neighbor_count[i] / area
    
    # Combine neighborhood features
    neighborhood_features[:, 0] = neighbor_count
    neighborhood_features[:, 1] = avg_distance
    neighborhood_features[:, 2] = density
    
    return neighborhood_features


def extract_features(nuclei: List[Dict], image: np.ndarray, feature_types: List[str] = None) -> Dict:
    """
    Extract features from detected nuclei.
    
    Args:
        nuclei: List of dictionaries containing nucleus information
        image: Original image
        feature_types: List of feature types to extract
        
    Returns:
        Dictionary with extracted features for each nucleus
    """
    if feature_types is None:
        feature_types = Config.FEATURE_TYPES
    
    num_nuclei = len(nuclei)
    
    # Initialize feature arrays
    geometric_features = np.zeros((num_nuclei, 9))
    intensity_features = np.zeros((num_nuclei, 5))
    texture_features = np.zeros((num_nuclei, 6))
    
    # Extract features for each nucleus
    for i, nucleus in enumerate(nuclei):
        if "geometric" in feature_types:
            geometric_features[i] = extract_geometric_features(nucleus)
        
        if "intensity" in feature_types:
            intensity_features[i] = extract_intensity_features(nucleus, image)
        
        if "texture" in feature_types:
            texture_features[i] = extract_texture_features(nucleus, image)
    
    # Extract neighborhood features
    if "neighborhood" in feature_types:
        neighborhood_features = extract_neighborhood_features(nuclei)
    else:
        neighborhood_features = np.zeros((num_nuclei, 3))
    
    # Combine all features
    all_features = np.hstack([
        geometric_features if "geometric" in feature_types else np.zeros((num_nuclei, 9)),
        intensity_features if "intensity" in feature_types else np.zeros((num_nuclei, 5)),
        texture_features if "texture" in feature_types else np.zeros((num_nuclei, 6)),
        neighborhood_features if "neighborhood" in feature_types else np.zeros((num_nuclei, 3))
    ])
    
    # Create result dictionary
    result = {
        'nuclei': nuclei,
        'features': {
            'geometric': geometric_features,
            'intensity': intensity_features,
            'texture': texture_features,
            'neighborhood': neighborhood_features,
            'all': all_features
        },
        'centroids': np.array([nucleus['centroid'] for nucleus in nuclei])
    }
    
    return result


class NucleiFeatureExtractor:
    """Feature extractor for nuclei in histology images."""
    
    def __init__(self, feature_types: List[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            feature_types: List of feature types to extract
        """
        self.feature_types = feature_types if feature_types else Config.FEATURE_TYPES
    
    def extract(self, nuclei: List[Dict], image: np.ndarray) -> Dict:
        """
        Extract features from nuclei.
        
        Args:
            nuclei: List of nuclei
            image: Original image
            
        Returns:
            Dictionary with extracted features
        """
        return extract_features(nuclei, image, self.feature_types)


if __name__ == "__main__":
    # Test with sample data
    import os
    import matplotlib.pyplot as plt
    from nuclei_detection import NucleiDetectionPipeline
    
    # Load and detect nuclei in a sample image
    sample_path = os.path.join(Config.DATA_ROOT, "sample.png")
    if os.path.exists(sample_path):
        # Load image
        image = cv2.imread(sample_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect nuclei
        pipeline = NucleiDetectionPipeline()
        detection_result = pipeline.detect(image, magnification=10)
        
        # Extract features
        feature_extractor = NucleiFeatureExtractor()
        features = feature_extractor.extract(detection_result['nuclei'], image)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(2, 2, 2)
        plt.scatter(
            features['centroids'][:, 0],
            features['centroids'][:, 1],
            c=features['features']['geometric'][:, 0],  # Color by area
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(label='Area')
        plt.title("Nuclei by Area")
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 3)
        plt.scatter(
            features['centroids'][:, 0],
            features['centroids'][:, 1],
            c=features['features']['neighborhood'][:, 1],  # Color by avg distance
            cmap='plasma',
            alpha=0.7
        )
        plt.colorbar(label='Avg Distance to Neighbors')
        plt.title("Nuclei by Neighborhood Distance")
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 4)
        plt.hist(features['features']['geometric'][:, 4], bins=20)  # Aspect ratio
        plt.title("Aspect Ratio Distribution")
        
        plt.tight_layout()
        plt.show()
