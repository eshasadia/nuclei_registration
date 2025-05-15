"""
Utility functions for nuclei registration model.
"""

import os
import torch
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from medpy.metric.binary import hd, dc


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
    """
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path, device=None):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load model onto
        
    Returns:
        epoch, val_loss
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and val_loss
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    return epoch, val_loss


def calculate_mse(image1, image2):
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        MSE value
    """
    # Convert to float
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    
    # Calculate MSE
    mse = np.mean((image1 - image2) ** 2)
    
    return mse


def calculate_ssim(image1, image2):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        SSIM value
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1_gray = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
    
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2_gray = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image2_gray = image2
    
    # Calculate SSIM
    ssim = structural_similarity(image1_gray, image2_gray)
    
    return ssim


def calculate_psnr(image1, image2):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        PSNR value
    """
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(image1, image2)
    
    return psnr


def calculate_dice_coefficient(image1, image2, threshold=127):
    """
    Calculate Dice coefficient between two images.
    
    Args:
        image1: First image
        image2: Second image
        threshold: Threshold for binary conversion
        
    Returns:
        Dice coefficient
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1_gray = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
    
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2_gray = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image2_gray = image2
    
    # Convert to binary
    binary1 = (image1_gray > threshold).astype(np.uint8)
    binary2 = (image2_gray > threshold).astype(np.uint8)
    
    # Calculate Dice coefficient
    dice = dc(binary1, binary2)
    
    return dice


def calculate_hausdorff_distance(image1, image2, threshold=127):
    """
    Calculate Hausdorff distance between two images.
    
    Args:
        image1: First image
        image2: Second image
        threshold: Threshold for binary conversion
        
    Returns:
        Hausdorff distance
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1_gray = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
    
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2_gray = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image2_gray = image2
    
    # Convert to binary
    binary1 = (image1_gray > threshold).astype(np.uint8)
    binary2 = (image2_gray > threshold).astype(np.uint8)
    
    try:
        # Calculate Hausdorff distance
        hausdorff = hd(binary1, binary2)
    except:
        # Return a large value if calculation fails
        hausdorff = 1000.0
    
    return hausdorff


def calculate_mean_nuclei_distance(fixed_nuclei, transformed_nuclei):
    """
    Calculate mean distance between corresponding nuclei.
    
    Args:
        fixed_nuclei: Nuclei in fixed image
        transformed_nuclei: Nuclei in transformed image
        
    Returns:
        Mean distance between corresponding nuclei
    """
    if len(fixed_nuclei) == 0 or len(transformed_nuclei) == 0:
        return float('inf')
    
    # Get centroids
    fixed_centroids = np.array([nucleus['centroid'] for nucleus in fixed_nuclei])
    transformed_centroids = np.array([nucleus['centroid'] for nucleus in transformed_nuclei])
    
    # Find nearest neighbors
    distances = []
    for centroid in fixed_centroids:
        # Find nearest transformed centroid
        dist = np.sqrt(np.sum((transformed_centroids - centroid) ** 2, axis=1))
        min_dist = np.min(dist)
        distances.append(min_dist)
    
    # Calculate mean distance
    mean_distance = np.mean(distances)
    
    return mean_distance


def calculate_target_registration_error(registration_result):
    """
    Calculate Target Registration Error (TRE) from registration result.
    
    Args:
        registration_result: Registration result dictionary
        
    Returns:
        Mean TRE value
    """
    # Get points
    if 'source_points' not in registration_result or 'target_points' not in registration_result:
        return float('inf')
    
    source_points = registration_result['source_points']
    target_points = registration_result['target_points']
    transformed_source = registration_result['transformed_source']
    
    # Create KD-tree for target points
    from scipy.spatial import KDTree
    target_tree = KDTree(target_points)
    
    # For each transformed source point, find the nearest target point
    distances, indices = target_tree.query(transformed_source)
    
    # TRE is the mean distance
    tre = np.mean(distances)
    
    return tre


def calculate_metrics(image1, image2, metrics=None):
    """
    Calculate multiple metrics between two images.
    
    Args:
        image1: First image
        image2: Second image
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of metric values
    """
    # Default metrics
    if metrics is None:
        metrics = ['mse', 'ssim', 'psnr']
    
    # Initialize results
    results = {}
    
    # Calculate each metric
    for metric in metrics:
        if metric == 'mse':
            results['mse'] = calculate_mse(image1, image2)
        elif metric == 'ssim':
            results['ssim'] = calculate_ssim(image1, image2)
        elif metric == 'psnr':
            results['psnr'] = calculate_psnr(image1, image2)
        elif metric == 'dice':
            results['dice'] = calculate_dice_coefficient(image1, image2)
        elif metric == 'hausdorff':
            results['hausdorff'] = calculate_hausdorff_distance(image1, image2)
        elif metric == 'tre' and isinstance(image2, dict) and 'registration_result' in image2:
            results['tre'] = calculate_target_registration_error(image2['registration_result'])
    
    return results


def create_evaluation_visualizations(fixed_image, moving_image, transformed_moving, registration_result=None):
    """
    Create visualization images for evaluation.
    
    Args:
        fixed_image: Fixed image
        moving_image: Moving image
        transformed_moving: Transformed moving image
        registration_result: Optional registration result dictionary
        
    Returns:
        Dictionary of visualization images
    """
    # Initialize results
    visualizations = {}
    
    # Add original images
    visualizations['fixed'] = fixed_image.copy()
    visualizations['moving'] = moving_image.copy()
    visualizations['transformed'] = transformed_moving.copy()
    
    # Create difference image
    diff_image = cv2.absdiff(fixed_image, transformed_moving)
    
    # Apply colormap for better visualization
    if len(diff_image.shape) == 3:
        diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_RGB2GRAY)
    else:
        diff_gray = diff_image
    
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    
    # Add difference image
    visualizations['difference'] = diff_color
    
    # Create overlay
    alpha = 0.5
    overlay = cv2.addWeighted(
        fixed_image,
        1 - alpha,
        transformed_moving,
        alpha,
        0
    )
    
    visualizations['overlay'] = overlay
    
    # Create checkerboard
    if fixed_image.shape == transformed_moving.shape:
        checkerboard = create_checkerboard(fixed_image, transformed_moving)
        visualizations['checkerboard'] = checkerboard
    
    # If registration result is provided, create point visualizations
    if registration_result is not None and 'registration_result' in registration_result:
        # Get registration result
        reg_result = registration_result['registration_result']
        
        # If point sets are available, visualize them
        if 'source_points' in reg_result and 'target_points' in reg_result:
            source_points = reg_result['source_points']
            target_points = reg_result['target_points']
            transformed_source = reg_result['transformed_source']
            
            # Create point visualizations
            points_vis = fixed_image.copy()
            
            # Draw target points (red)
            for point in target_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(points_vis, (x, y), 3, (255, 0, 0), -1)
            
            # Draw transformed source points (green)
            for point in transformed_source:
                x, y = int(point[0]), int(point[1])
                cv2.circle(points_vis, (x, y), 3, (0, 255, 0), -1)
            
            visualizations['points'] = points_vis
    
    return visualizations


def create_checkerboard(image1, image2, tile_size=50):
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


def create_grid_overlay(image, grid_size=50, color=(0, 255, 0), thickness=1):
    """
    Create a grid overlay on an image.
    
    Args:
        image: Input image
        grid_size: Size of grid cells
        color: Color of grid lines (BGR)
        thickness: Thickness of grid lines
        
    Returns:
        Image with grid overlay
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Draw horizontal lines
    for i in range(0, h, grid_size):
        cv2.line(result, (0, i), (w, i), color, thickness)
    
    # Draw vertical lines
    for j in range(0, w, grid_size):
        cv2.line(result, (j, 0), (j, h), color, thickness)
    
    return result


def estimate_magnification(image, nuclei_detector=None):
    """
    Estimate magnification from image based on nuclei size and density.
    
    Args:
        image: Input histology image
        nuclei_detector: Optional nuclei detector
        
    Returns:
        Estimated magnification (5, 10, 20, or 40)
    """
    if nuclei_detector is None:
        # Simple estimation based on image features
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size
        min_area = 20
        max_area = 500
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        # Calculate average contour area
        if len(valid_contours) > 0:
            avg_area = np.mean([cv2.contourArea(c) for c in valid_contours])
            
            # Estimate magnification based on area
            if avg_area > 300:
                magnification = 5
            elif avg_area > 150:
                magnification = 10
            elif avg_area > 80:
                magnification = 20
            else:
                magnification = 40
        else:
            # Default if no contours found
            magnification = 20
    else:
        # Use nuclei detector to get more accurate estimate
        detection_result = nuclei_detector.detect(image)
        
        # Get nuclei
        nuclei = detection_result['nuclei']
        
        if len(nuclei) > 0:
            # Calculate average nucleus area
            avg_area = np.mean([nucleus['area'] for nucleus in nuclei])
            
            # Estimate magnification based on area
            if avg_area > 300:
                magnification = 5
            elif avg_area > 150:
                magnification = 10
            elif avg_area > 80:
                magnification = 20
            else:
                magnification = 40
        else:
            # Default if no nuclei found
            magnification = 20
    
    return magnification


def create_composite_visualization(registration_result):
    """
    Create a composite visualization of registration results.
    
    Args:
        registration_result: Registration result dictionary
        
    Returns:
        Composite visualization image
    """
    # Extract images
    fixed_image = registration_result['fixed_image']
    moving_image = registration_result['moving_image']
    transformed_moving = registration_result['transformed_moving']
    
    # Create visualizations
    visualizations = create_evaluation_visualizations(
        fixed_image,
        moving_image,
        transformed_moving,
        registration_result
    )
    
    # Create composite image
    h, w = fixed_image.shape[:2]
    
    # Create a 2x3 grid
    composite = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    
    # Add images to grid
    composite[:h, :w] = visualizations['fixed']
    composite[:h, w:2*w] = visualizations['moving']
    composite[:h, 2*w:] = visualizations['transformed']
    composite[h:, :w] = visualizations['difference']
    composite[h:, w:2*w] = visualizations['overlay']
    
    if 'checkerboard' in visualizations:
        composite[h:, 2*w:] = visualizations['checkerboard']
    elif 'points' in visualizations:
        composite[h:, 2*w:] = visualizations['points']
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    font_thickness = 2
    bg_color = (0, 0, 0)
    padding = 5
    
    labels = [
        "Fixed Image", "Moving Image", "Transformed Moving",
        "Difference", "Overlay", "Checkerboard/Points"
    ]
    
    for i, label in enumerate(labels):
        row = i // 3
        col = i % 3
        
        # Prepare text
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Position for text
        x = col * w + (w - text_width) // 2
        y = row * h + 30
        
        # Draw background rectangle
        cv2.rectangle(
            composite,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            composite,
            label,
            (x, y),
            font,
            font_scale,
            font_color,
            font_thickness
        )
    
    # Add metrics
    metrics = calculate_metrics(
        fixed_image,
        transformed_moving,
        ['mse', 'ssim', 'psnr']
    )
    
    metrics_text = f"MSE: {metrics['mse']:.2f} | SSIM: {metrics['ssim']:.2f} | PSNR: {metrics['psnr']:.2f}"
    
    # Prepare text
    (text_width, text_height), _ = cv2.getTextSize(
        metrics_text, font, font_scale, font_thickness
    )
    
    # Position for text
    x = (composite.shape[1] - text_width) // 2
    y = composite.shape[0] - 20
    
    # Draw background rectangle
    cv2.rectangle(
        composite,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        composite,
        metrics_text,
        (x, y),
        font,
        font_scale,
        font_color,
        font_thickness
    )
    
    return composite
