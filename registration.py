"""
Registration module implementing various point-set registration methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import cv2
from scipy.spatial import KDTree
from config import Config


def icp_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    initial_transform: np.ndarray = None
) -> Dict:
    """
    Iterative Closest Point (ICP) algorithm for rigid registration.
    
    Args:
        source_points: Source points (N x 2)
        target_points: Target points (M x 2)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        initial_transform: Initial transformation matrix (3 x 3)
        
    Returns:
        Dictionary with registration results
    """
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("At least 3 points required for ICP")
    
    # Convert to homogeneous coordinates
    source_homog = np.hstack([source_points, np.ones((len(source_points), 1))])
    
    # Initialize transformation
    if initial_transform is None:
        current_transform = np.eye(3)
    else:
        current_transform = initial_transform.copy()
    
    # Initialize convergence parameters
    prev_error = float('inf')
    
    # Create KD-tree for fast nearest neighbor lookup
    target_tree = KDTree(target_points)
    
    # Keep track of all transformations
    transforms = [current_transform.copy()]
    errors = []
    
    # ICP iterations
    for iteration in range(max_iterations):
        # Apply current transformation to source points
        transformed_source = np.dot(source_homog, current_transform.T)[:, :2]
        
        # Find closest points in target
        distances, indices = target_tree.query(transformed_source)
        corresponding_target = target_points[indices]
        
        # Calculate mean squared error
        error = np.mean(distances ** 2)
        errors.append(error)
        
        # Check for convergence
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error
        
        # Compute optimal rotation and translation
        source_centered = transformed_source - np.mean(transformed_source, axis=0)
        target_centered = corresponding_target - np.mean(corresponding_target, axis=0)
        
        # Covariance matrix
        H = source_centered.T @ target_centered
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Rotation matrix
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation vector
        t = np.mean(corresponding_target, axis=0) - np.mean(transformed_source @ R, axis=0)
        
        # Update transformation matrix
        update = np.eye(3)
        update[:2, :2] = R
        update[:2, 2] = t
        
        current_transform = update @ current_transform
        transforms.append(current_transform.copy())
    
    # Apply final transformation to source points
    final_source = np.dot(source_homog, current_transform.T)[:, :2]
    
    # Calculate final mean squared error
    distances, _ = target_tree.query(final_source)
    final_error = np.mean(distances ** 2)
    
    # Return registration results
    return {
        'source_points': source_points,
        'target_points': target_points,
        'transformed_source': final_source,
        'transformation': current_transform,
        'iterations': len(transforms),
        'errors': errors,
        'final_error': final_error
    }


def ransac_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray = None,
    target_features: np.ndarray = None,
    max_iterations: int = 1000,
    distance_threshold: float = 10.0,
    min_inliers: int = 3
) -> Dict:
    """
    RANSAC algorithm for robust point set registration.
    
    Args:
        source_points: Source points (N x 2)
        target_points: Target points (M x 2)
        source_features: Features for source points (N x F)
        target_features: Features for target points (M x F)
        max_iterations: Maximum number of RANSAC iterations
        distance_threshold: Maximum distance for inlier classification
        min_inliers: Minimum number of inliers for a valid model
        
    Returns:
        Dictionary with registration results
    """
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("At least 3 points required for RANSAC")
    
    # If features are provided, use them for better correspondence matching
    if source_features is not None and target_features is not None:
        # Normalize features for better matching
        source_features_norm = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-10)
        target_features_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-10)
        
        # Compute feature similarity matrix
        similarity = source_features_norm @ target_features_norm.T
        
        # Find potential correspondences (for each source point, get top 3 target candidates)
        k = min(3, target_points.shape[0])
        correspondence_candidates = np.argsort(-similarity, axis=1)[:, :k]
    else:
        # Without features, use proximity-based correspondence candidates
        target_tree = KDTree(target_points)
        _, correspondence_candidates = target_tree.query(source_points, k=min(3, target_points.shape[0]))
    
    best_inliers = []
    best_transform = np.eye(3)
    best_error = float('inf')
    
    # RANSAC iterations
    for i in range(max_iterations):
        # Randomly select 3 source points
        idx = np.random.choice(len(source_points), 3, replace=False)
        sample_source = source_points[idx]
        
        # For each source point, randomly select one of its candidate correspondences
        sample_target_idx = []
        for j in idx:
            candidate_idx = correspondence_candidates[j]
            selected_idx = np.random.choice(candidate_idx)
            sample_target_idx.append(selected_idx)
        
        sample_target = target_points[sample_target_idx]
        
        # Compute transformation from sample points
        try:
            # Compute rigid transformation
            source_centered = sample_source - np.mean(sample_source, axis=0)
            target_centered = sample_target - np.mean(sample_target, axis=0)
            
            H = source_centered.T @ target_centered
            U, _, Vt = np.linalg.svd(H)
            
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = np.mean(sample_target, axis=0) - np.mean(sample_source @ R, axis=0)
            
            # Create transformation matrix
            transform = np.eye(3)
            transform[:2, :2] = R
            transform[:2, 2] = t
        except np.linalg.LinAlgError:
            continue
        
        # Apply transformation to all source points
        source_homog = np.hstack([source_points, np.ones((len(source_points), 1))])
        transformed_source = np.dot(source_homog, transform.T)[:, :2]
        
        # Find closest target points
        target_tree = KDTree(target_points)
        distances, closest_idx = target_tree.query(transformed_source)
        
        # Find inliers
        inliers = np.where(distances < distance_threshold)[0]
        
        # If we have enough inliers, refine the transformation
        if len(inliers) >= min_inliers:
            # Refine transformation using all inliers
            inlier_source = source_points[inliers]
            inlier_target = target_points[closest_idx[inliers]]
            
            # Refine transformation
            try:
                source_centered = inlier_source - np.mean(inlier_source, axis=0)
                target_centered = inlier_target - np.mean(inlier_target, axis=0)
                
                H = source_centered.T @ target_centered
                U, _, Vt = np.linalg.svd(H)
                
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                t = np.mean(inlier_target, axis=0) - np.mean(inlier_source @ R, axis=0)
                
                refined_transform = np.eye(3)
                refined_transform[:2, :2] = R
                refined_transform[:2, 2] = t
                
                # Apply refined transformation
                transformed_source = np.dot(source_homog, refined_transform.T)[:, :2]
                
                # Recalculate distances and errors
                distances, _ = target_tree.query(transformed_source)
                error = np.mean(distances ** 2)
                
                # Update best model if this one is better
                if error < best_error:
                    best_error = error
                    best_transform = refined_transform
                    best_inliers = inliers
            except np.linalg.LinAlgError:
                continue
    
    # No valid model found
    if len(best_inliers) < min_inliers:
        return {
            'source_points': source_points,
            'target_points': target_points,
            'transformed_source': source_points,  # No transformation
            'transformation': np.eye(3),
            'inliers': [],
            'success': False,
            'error': float('inf')
        }
    
    # Apply best transformation to source points
    source_homog = np.hstack([source_points, np.ones((len(source_points), 1))])
    final_source = np.dot(source_homog, best_transform.T)[:, :2]
    
    # Calculate final error
    distances, _ = KDTree(target_points).query(final_source)
    final_error = np.mean(distances ** 2)
    
    return {
        'source_points': source_points,
        'target_points': target_points,
        'transformed_source': final_source,
        'transformation': best_transform,
        'inliers': best_inliers,
        'success': True,
        'error': final_error
    }


def cpd_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    w: float = 0.0,
    beta: float = 2.0,
    lambda_param: float = 3.0,
    rigid: bool = True
) -> Dict:
    """
    Coherent Point Drift (CPD) algorithm for non-rigid point set registration.
    
    Args:
        source_points: Source points (N x 2)
        target_points: Target points (M x 2)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        w: Weight of the uniform distribution (0 <= w < 1)
        beta: Width of the Gaussian filter
        lambda_param: Regularization weight
        rigid: Whether to use rigid or non-rigid registration
        
    Returns:
        Dictionary with registration results
    """
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("At least 3 points required for CPD")
    
    # Initialize variables
    N, D = source_points.shape
    M, _ = target_points.shape
    
    # Center the point sets
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    
    # Scale the point sets
    source_scale = np.sqrt(np.sum(source_centered ** 2) / N)
    target_scale = np.sqrt(np.sum(target_centered ** 2) / M)
    
    source_centered /= source_scale
    target_centered /= target_scale
    
    # Initialize transformation
    sigma2 = 0.5 * (M * np.trace(target_centered.T @ target_centered) +
                   N * np.trace(source_centered.T @ source_centered)) / (M * N)
    
    if rigid:
        # Rigid transformation parameters
        R = np.eye(D)
        t = np.zeros(D)
        s = 1.0
    else:
        # Non-rigid transformation parameters
        G = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist_sq = np.sum((source_centered[i] - source_centered[j]) ** 2)
                G[i, j] = np.exp(-dist_sq / (2 * beta ** 2))
        
        W = np.zeros((N, D))
    
    # Initialize variables for EM algorithm
    P = np.zeros((M, N))
    prev_error = float('inf')
    iterations = 0
    
    # EM optimization
    for iteration in range(max_iterations):
        iterations += 1
        
        # E-step: compute posterior probability
        if rigid:
            # Apply current transformation
            transformed_source = s * source_centered @ R.T + t
        else:
            # Apply current transformation
            transformed_source = source_centered + G @ W
        
        # Compute distances between all pairs of points
        distances = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                distances[i, j] = np.sum((target_centered[i] - transformed_source[j]) ** 2)
        
        # Compute posterior probability matrix
        c = (2 * np.pi * sigma2) ** (D / 2) * w * (1 - w) / ((1 - w) * M)
        P = np.exp(-distances / (2 * sigma2))
        den = np.sum(P, axis=1, keepdims=True) + c
        P = P / den
        
        # Compute error
        error = np.sum(distances * P) + D * N * np.log(sigma2) / 2
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error
        
        # M-step: update transformation
        Np = np.sum(P)
        target_center = P.T @ target_centered / Np
        source_center = source_centered
        
        if rigid:
            # Update rigid transformation
            A = source_centered.T @ P.T @ target_centered
            U, _, Vt = np.linalg.svd(A)
            
            # Handle reflection case
            C = np.eye(D)
            if np.linalg.det(Vt.T @ U.T) < 0:
                C[-1, -1] = -1
            
            R = Vt.T @ C @ U.T
            
            # Update scale and translation
            s = np.trace(A @ R) / np.trace(source_centered.T @ np.diag(np.sum(P, axis=0)) @ source_centered)
            t = target_center - s * source_center @ R.T
            
            # Update sigma2
            transformed_source = s * source_centered @ R.T + t
        else:
            # Update non-rigid transformation
            dP = np.diag(np.sum(P, axis=0))
            F = np.linalg.inv(G + lambda_param * sigma2 * np.linalg.inv(dP) @ G)
            W = F @ P.T @ target_centered
            
            # Update sigma2
            transformed_source = source_centered + G @ W
        
        # Recompute sigma2
        sigma2_new = 0
        for i in range(M):
            for j in range(N):
                sigma2_new += P[i, j] * np.sum((target_centered[i] - transformed_source[j]) ** 2)
        
        sigma2 = sigma2_new / (Np * D)
        
        # Avoid numerical issues
        if sigma2 < 1e-10:
            sigma2 = 1e-10
    
    # Apply final transformation to original source points
    if rigid:
        # Un-normalize transformation
        R_final = R
        s_final = s * target_scale / source_scale
        t_final = target_mean - s_final * source_mean @ R_final.T
        
        # Create transformation matrix
        transformation = np.eye(3)
        transformation[:2, :2] = s_final * R_final
        transformation[:2, 2] = t_final
        
        # Apply transformation
        source_homog = np.hstack([source_points, np.ones((N, 1))])
        transformed_source = np.dot(source_homog, transformation.T)[:, :2]
        
        result = {
            'source_points': source_points,
            'target_points': target_points,
            'transformed_source': transformed_source,
            'transformation': transformation,
            'rotation': R_final,
            'scale': s_final,
            'translation': t_final,
            'iterations': iterations,
            'error': error,
            'correspondence': P
        }
    else:
        # Un-normalize transformation
        source_normalized = (source_points - source_mean) / source_scale
        target_normalized = (target_points - target_mean) / target_scale
        
        # Compute G matrix for original points
        G_original = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist_sq = np.sum((source_normalized[i] - source_normalized[j]) ** 2)
                G_original[i, j] = np.exp(-dist_sq / (2 * beta ** 2))
        
        # Apply non-rigid transformation
        transformed_source_normalized = source_normalized + G_original @ W
        transformed_source = transformed_source_normalized * target_scale + target_mean
        
        result = {
            'source_points': source_points,
            'target_points': target_points,
            'transformed_source': transformed_source,
            'transformation': None,  # No simple matrix for non-rigid
            'displacement_field': W,
            'iterations': iterations,
            'error': error,
            'correspondence': P
        }
    
    return result


class RegistrationModule:
    """Module for point-set registration."""
    
    def __init__(self, method: str = None):
        """
        Initialize the registration module.
        
        Args:
            method: Registration method ('icp', 'ransac', or 'cpd')
        """
        self.method = method if method else Config.REGISTRATION_METHOD
    
    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        source_features: np.ndarray = None,
        target_features: np.ndarray = None
    ) -> Dict:
        """
        Register source points to target points.
        
        Args:
            source_points: Source points (N x 2)
            target_points: Target points (M x 2)
            source_features: Features for source points (N x F)
            target_features: Features for target points (M x F)
            
        Returns:
            Registration results
        """
        # Convert to numpy arrays if needed
        if isinstance(source_points, torch.Tensor):
            source_points = source_points.detach().cpu().numpy()
        if isinstance(target_points, torch.Tensor):
            target_points = target_points.detach().cpu().numpy()
        if isinstance(source_features, torch.Tensor):
            source_features = source_features.detach().cpu().numpy()
        if isinstance(target_features, torch.Tensor):
            target_features = target_features.detach().cpu().numpy()
        
        # Apply registration method
        if self.method == 'icp':
            return icp_registration(
                source_points,
                target_points,
                max_iterations=Config.MAX_ITERATIONS,
                tolerance=Config.CONVERGENCE_THRESHOLD
            )
        elif self.method == 'ransac':
            return ransac_registration(
                source_points,
                target_points,
                source_features,
                target_features,
                max_iterations=Config.MAX_ITERATIONS
            )
        elif self.method == 'cpd':
            return cpd_registration(
                source_points,
                target_points,
                max_iterations=Config.MAX_ITERATIONS,
                tolerance=Config.CONVERGENCE_THRESHOLD,
                w=Config.OUTLIER_WEIGHT,
                lambda_param=Config.REGULARIZATION_WEIGHT
            )
        else:
            raise ValueError(f"Unknown registration method: {self.method}")
    
    def transform_image(self, image: np.ndarray, transformation: np.ndarray) -> np.ndarray:
        """
        Apply transformation to an image.
        
        Args:
            image: Input image
            transformation: Transformation matrix (3 x 3)
            
        Returns:
            Transformed image
        """
        h, w = image.shape[:2]
        
        # Apply transformation
        return cv2.warpAffine(
            image,
            transformation[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR
        )


def visualize_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    registration_result: Dict,
    overlay_alpha: float = 0.5
) -> Dict:
    """
    Visualize registration results.
    
    Args:
        fixed_image: Fixed image
        moving_image: Moving image
        registration_result: Registration results
        overlay_alpha: Alpha value for overlay blend
        
    Returns:
        Dictionary with visualizations
    """
    # Get transformation
    transformation = registration_result.get('transformation')
    
    # Initialize visualizations
    visualizations = {}
    
    # Transformed source points
    source_points = registration_result['source_points']
    target_points = registration_result['target_points']
    transformed_points = registration_result['transformed_source']
    
    # Create point visualizations
    h, w = fixed_image.shape[:2]
    
    # Source points on moving image
    moving_points = moving_image.copy()
    for point in source_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(moving_points, (x, y), 3, (0, 255, 0), -1)
    
    # Target points on fixed image
    fixed_points = fixed_image.copy()
    for point in target_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(fixed_points, (x, y), 3, (0, 0, 255), -1)
    
    # Transformed source points on fixed image
    transformed_vis = fixed_image.copy()
    for point in transformed_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(transformed_vis, (x, y), 3, (255, 0, 0), -1)
    
    # Generate transformed moving image if transformation is available
    if transformation is not None:
        # Apply transformation to moving image
        transformed_moving = cv2.warpAffine(
            moving_image,
            transformation[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        
        # Create checkerboard visualization
        checkerboard = create_checkerboard(fixed_image, transformed_moving)
        
        # Create overlay visualization
        overlay = cv2.addWeighted(
            fixed_image,
            1 - overlay_alpha,
            transformed_moving,
            overlay_alpha,
            0
        )
        
        # Add to visualizations
        visualizations['transformed_moving'] = transformed_moving
        visualizations['checkerboard'] = checkerboard
        visualizations['overlay'] = overlay
    
    # Add basic visualizations
    visualizations['fixed_image'] = fixed_image
    visualizations['moving_image'] = moving_image
    visualizations['fixed_points'] = fixed_points
    visualizations['moving_points'] = moving_points
    visualizations['transformed_points'] = transformed_vis
    
    return visualizations


def create_checkerboard(image1: np.ndarray, image2: np.ndarray, tile_size: int = 50) -> np.ndarray:
    """
    Create a checkerboard visualization of two images.
    
    Args:
        image1: First image
        image2: Second image
        tile_size: Size of checkerboard tiles
        
    Returns:
        Checkerboard image
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")
    
    h, w = image1.shape[:2]
    result = image1.copy()
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 1:
                # Get tile boundaries
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                
                # Copy tile from image2
                result[i:i_end, j:j_end] = image2[i:i_end, j:j_end]
    
    return result


if __name__ == "__main__":
    # Test the registration module
    import os
    import matplotlib.pyplot as plt
    from nuclei_detection import NucleiDetectionPipeline
    from feature_extraction import NucleiFeatureExtractor
    
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
        
        # Detect nuclei
        detector = NucleiDetectionPipeline()
        
        detection1 = detector.detect(image1)
        detection2 = detector.detect(image2)
        
        # Extract features
        feature_extractor = NucleiFeatureExtractor()
        
        features1 = feature_extractor.extract(detection1['nuclei'], image1)
        features2 = feature_extractor.extract(detection2['nuclei'], image2)
        
        # Register points
        registration = RegistrationModule()
        
        result = registration.register(
            features1['centroids'],
            features2['centroids'],
            features1['features']['all'],
            features2['features']['all']
        )
        
        # Visualize results
        vis = visualize_registration(image2, image1, result)
        
        # Display
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(vis['moving_image'])
        plt.title("Moving Image")
        
        plt.subplot(2, 3, 2)
        plt.imshow(vis['fixed_image'])
        plt.title("Fixed Image")
        
        plt.subplot(2, 3, 3)
        plt.imshow(vis['moving_points'])
        plt.title("Source Points")
        
        plt.subplot(2, 3, 4)
        plt.imshow(vis['fixed_points'])
        plt.title("Target Points")
        
        plt.subplot(2, 3, 5)
        plt.imshow(vis['transformed_points'])
        plt.title("Transformed Points")
        
        if 'overlay' in vis:
            plt.subplot(2, 3, 6)
            plt.imshow(vis['overlay'])
            plt.title("Overlay")
        
        plt.tight_layout()
        plt.show()
