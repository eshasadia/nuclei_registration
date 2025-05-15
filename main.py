"""
Main script for the nuclei registration system.
"""

import os
import argparse
import logging
import datetime
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import Config
from model import NucleiRegistrationModel
from utils import create_composite_visualization, estimate_magnification
from nuclei_detection import NucleiDetectionPipeline
from feature_extraction import NucleiFeatureExtractor
from registration import RegistrationModule


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'main.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def register_images(args):
    """
    Register two images.
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Log arguments
    logger.info("Registration arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if input images exist
    if not os.path.exists(args.fixed_image):
        logger.error(f"Fixed image not found: {args.fixed_image}")
        return
    
    if not os.path.exists(args.moving_image):
        logger.error(f"Moving image not found: {args.moving_image}")
        return
    
    # Load images
    fixed_image = cv2.imread(args.fixed_image)
    fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB)
    
    moving_image = cv2.imread(args.moving_image)
    moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = NucleiRegistrationModel(
        detector_weights_path=args.model_path,
        registration_method=args.registration_method
    )
    
    # Estimate magnification if not provided
    fixed_magnification = args.fixed_magnification
    moving_magnification = args.moving_magnification
    
    if fixed_magnification is None:
        fixed_magnification = estimate_magnification(fixed_image)
        logger.info(f"Estimated fixed image magnification: {fixed_magnification}x")
    
    if moving_magnification is None:
        moving_magnification = estimate_magnification(moving_image)
        logger.info(f"Estimated moving image magnification: {moving_magnification}x")
    
    # Register images
    logger.info("Registering images...")
    
    try:
        result = model.register_images(
            fixed_image,
            moving_image,
            fixed_magnification=fixed_magnification,
            moving_magnification=moving_magnification
        )
        
        # Get transformed image
        transformed_moving = result['transformed_moving']
        
        # Save results
        logger.info("Saving results...")
        
        # Save transformed image
        transformed_path = os.path.join(args.output_dir, 'transformed.png')
        cv2.imwrite(transformed_path, cv2.cvtColor(transformed_moving, cv2.COLOR_RGB2BGR))
        
        # Save visualization
        composite = create_composite_visualization(result)
        composite_path = os.path.join(args.output_dir, 'visualization.png')
        cv2.imwrite(composite_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        
        # Save individual visualizations
        if args.save_all:
            for name, image in result['visualizations'].items():
                path = os.path.join(args.output_dir, f'{name}.png')
                cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved results to {args.output_dir}")
        
        # Display results if requested
        if args.display:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(fixed_image)
            plt.title("Fixed Image")
            
            plt.subplot(2, 3, 2)
            plt.imshow(moving_image)
            plt.title("Moving Image")
            
            plt.subplot(2, 3, 3)
            plt.imshow(transformed_moving)
            plt.title("Transformed Moving")
            
            plt.subplot(2, 3, 4)
            plt.imshow(result['visualizations']['difference'])
            plt.title("Difference")
            
            plt.subplot(2, 3, 5)
            plt.imshow(result['visualizations']['overlay'])
            plt.title("Overlay")
            
            if 'checkerboard' in result['visualizations']:
                plt.subplot(2, 3, 6)
                plt.imshow(result['visualizations']['checkerboard'])
                plt.title("Checkerboard")
            elif 'points' in result['visualizations']:
                plt.subplot(2, 3, 6)
                plt.imshow(result['visualizations']['points'])
                plt.title("Points")
            
            plt.tight_layout()
            plt.show()
        
        return result
    
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_directory(args):
    """
    Process a directory of image pairs.
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Log arguments
    logger.info("Directory processing arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find image pairs
    image_pairs = []
    
    # Try to find pairs with naming convention
    import glob
    
    # Look for pairs with _fixed and _moving suffixes
    fixed_images = sorted(glob.glob(os.path.join(args.input_dir, '*_fixed.*')))
    moving_images = sorted(glob.glob(os.path.join(args.input_dir, '*_moving.*')))
    
    if len(fixed_images) > 0 and len(moving_images) > 0:
        # Match by name
        fixed_bases = [os.path.splitext(os.path.basename(f))[0].replace('_fixed', '') for f in fixed_images]
        moving_bases = [os.path.splitext(os.path.basename(m))[0].replace('_moving', '') for m in moving_images]
        
        # Find matching bases
        for i, fixed_base in enumerate(fixed_bases):
            if fixed_base in moving_bases:
                j = moving_bases.index(fixed_base)
                image_pairs.append((fixed_images[i], moving_images[j]))
    
    # If no pairs found, try to pair sequentially
    if len(image_pairs) == 0:
        all_images = sorted(
            glob.glob(os.path.join(args.input_dir, '*.png')) +
            glob.glob(os.path.join(args.input_dir, '*.jpg')) +
            glob.glob(os.path.join(args.input_dir, '*.tif')) +
            glob.glob(os.path.join(args.input_dir, '*.tiff'))
        )
        
        if len(all_images) % 2 == 0:
            for i in range(0, len(all_images), 2):
                image_pairs.append((all_images[i], all_images[i+1]))
    
    if len(image_pairs) == 0:
        logger.error(f"No image pairs found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(image_pairs)} image pairs")
    
    # Initialize model
    model = NucleiRegistrationModel(
        detector_weights_path=args.model_path,
        registration_method=args.registration_method
    )
    
    # Process each pair
    for i, (fixed_path, moving_path) in enumerate(image_pairs):
        logger.info(f"Processing pair {i+1}/{len(image_pairs)}: {os.path.basename(fixed_path)} - {os.path.basename(moving_path)}")
        
        # Create pair output directory
        pair_dir = os.path.join(args.output_dir, f"pair_{i+1:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Set up arguments for registration
        pair_args = argparse.Namespace(
            fixed_image=fixed_path,
            moving_image=moving_path,
            output_dir=pair_dir,
            model_path=args.model_path,
            registration_method=args.registration_method,
            fixed_magnification=args.fixed_magnification,
            moving_magnification=args.moving_magnification,
            save_all=args.save_all,
            display=False
        )
        
        # Register images
        register_images(pair_args)
    
    logger.info(f"Processed {len(image_pairs)} image pairs")


def detect_nuclei(args):
    """
    Detect nuclei in an image.
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Log arguments
    logger.info("Nuclei detection arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Estimate magnification if not provided
    magnification = args.magnification
    if magnification is None:
        magnification = estimate_magnification(image)
        logger.info(f"Estimated image magnification: {magnification}x")
    
    # Initialize detector
    detector = NucleiDetectionPipeline(model_path=args.model_path)
    
    # Detect nuclei
    logger.info("Detecting nuclei...")
    detection_result = detector.detect(image, magnification=magnification)
    
    # Visualize detections
    logger.info(f"Detected {detection_result['count']} nuclei")
    
    # Create visualization
    vis_image = detector.visualize_detections(detection_result, show_centroids=True)
    
    # Save visualization
    vis_path = os.path.join(args.output_dir, 'nuclei_detection.png')
    cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Save detection data
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_safe_nuclei = []
    for nucleus in detection_result['nuclei']:
        json_safe_nucleus = {
            'id': nucleus['id'],
            'centroid': nucleus['centroid'].tolist(),
            'x': int(nucleus['x']),
            'y': int(nucleus['y']),
            'width': int(nucleus['width']),
            'height': int(nucleus['height']),
            'area': float(nucleus['area']),
            'eccentricity': float(nucleus['eccentricity'])
            # Exclude mask as it's a large numpy array
        }
        json_safe_nuclei.append(json_safe_nucleus)
    
    json_data = {
        'count': detection_result['count'],
        'nuclei': json_safe_nuclei,
        'magnification': magnification
    }
    
    json_path = os.path.join(args.output_dir, 'nuclei_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Saved results to {args.output_dir}")
    
    # Display results if requested
    if args.display:
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(vis_image)
        plt.title(f"Detected Nuclei: {detection_result['count']}")
        
        plt.tight_layout()
        plt.show()
    
    return detection_result


def extract_features(args):
    """
    Extract features from nuclei in an image.
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Log arguments
    logger.info("Feature extraction arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Estimate magnification if not provided
    magnification = args.magnification
    if magnification is None:
        magnification = estimate_magnification(image)
        logger.info(f"Estimated image magnification: {magnification}x")
    
    # Initialize detector
    detector = NucleiDetectionPipeline(model_path=args.model_path)
    
    # Detect nuclei
    logger.info("Detecting nuclei...")
    detection_result = detector.detect(image, magnification=magnification)
    
    # Initialize feature extractor
    logger.info("Extracting features...")
    feature_extractor = NucleiFeatureExtractor()
    features = feature_extractor.extract(detection_result['nuclei'], image)
    
    # Save features
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_safe_features = {
        'centroids': features['centroids'].tolist(),
        'features': {
            'geometric': features['features']['geometric'].tolist(),
            'intensity': features['features']['intensity'].tolist(),
            'texture': features['features']['texture'].tolist(),
            'neighborhood': features['features']['neighborhood'].tolist()
        }
    }
    
    json_path = os.path.join(args.output_dir, 'features.json')
    with open(json_path, 'w') as f:
        json.dump(json_safe_features, f, indent=2)
    
    # Create visualization
    logger.info("Creating visualizations...")
    
    # Visualize nuclei
    vis_image = detector.visualize_detections(detection_result, show_centroids=True)
    
    # Visualize features
    feature_vis = np.zeros_like(image)
    
    # Get a feature to visualize (e.g., area)
    if features['features']['geometric'].size > 0:
        areas = features['features']['geometric'][:, 0]  # Areas
        
        # Normalize areas for visualization
        if np.max(areas) > np.min(areas):
            normalized_areas = (areas - np.min(areas)) / (np.max(areas) - np.min(areas))
        else:
            normalized_areas = np.ones_like(areas) * 0.5
        
        # Draw nuclei with colors based on area
        for i, nucleus in enumerate(detection_result['nuclei']):
            x, y = int(nucleus['centroid'][0]), int(nucleus['centroid'][1])
            
            # Calculate color based on normalized area
            color = plt.cm.viridis(normalized_areas[i])[:3]
            color = (color[0] * 255, color[1] * 255, color[2] * 255)
            
            # Draw circle with size based on area
            radius = int(np.sqrt(nucleus['area'] / np.pi))
            cv2.circle(feature_vis, (x, y), radius, color, -1)
    
    # Save visualizations
    cv2.imwrite(
        os.path.join(args.output_dir, 'nuclei_detection.png'),
        cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    )
    
    cv2.imwrite(
        os.path.join(args.output_dir, 'feature_visualization.png'),
        cv2.cvtColor(feature_vis, cv2.COLOR_RGB2BGR)
    )
    
    logger.info(f"Saved results to {args.output_dir}")
    
    # Display results if requested
    if args.display:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(vis_image)
        plt.title(f"Detected Nuclei: {detection_result['count']}")
        
        plt.subplot(1, 3, 3)
        plt.imshow(feature_vis)
        plt.title("Feature Visualization (Area)")
        
        plt.tight_layout()
        plt.show()
    
    return features


def run_pipeline():
    """Run the main pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nuclei Registration System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register two images')
    register_parser.add_argument('--fixed_image', required=True, help='Path to fixed image')
    register_parser.add_argument('--moving_image', required=True, help='Path to moving image')
    register_parser.add_argument('--output_dir', default=os.path.join(Config.OUTPUT_DIR, 'registration'),
                              help='Output directory')
    register_parser.add_argument('--model_path', default=None, help='Path to trained model weights')
    register_parser.add_argument('--registration_method', default=Config.REGISTRATION_METHOD,
                              choices=['icp', 'ransac', 'cpd'], help='Registration method')
    register_parser.add_argument('--fixed_magnification', type=int, default=None,
                              help='Magnification of fixed image')
    register_parser.add_argument('--moving_magnification', type=int, default=None,
                              help='Magnification of moving image')
    register_parser.add_argument('--save_all', action='store_true', help='Save all visualizations')
    register_parser.add_argument('--display', action='store_true', help='Display results')
    
    # Process directory command
    process_parser = subparsers.add_parser('process', help='Process a directory of image pairs')
    process_parser.add_argument('--input_dir', required=True, help='Input directory')
    process_parser.add_argument('--output_dir', default=os.path.join(Config.OUTPUT_DIR, 'batch_registration'),
                             help='Output directory')
    process_parser.add_argument('--model_path', default=None, help='Path to trained model weights')
    process_parser.add_argument('--registration_method', default=Config.REGISTRATION_METHOD,
                             choices=['icp', 'ransac', 'cpd'], help='Registration method')
    process_parser.add_argument('--fixed_magnification', type=int, default=None,
                             help='Magnification of fixed images')
    process_parser.add_argument('--moving_magnification', type=int, default=None,
                             help='Magnification of moving images')
    process_parser.add_argument('--save_all', action='store_true', help='Save all visualizations')
    
    # Detect nuclei command
    detect_parser = subparsers.add_parser('detect', help='Detect nuclei in an image')
    detect_parser.add_argument('--input_image', required=True, help='Path to input image')
    detect_parser.add_argument('--output_dir', default=os.path.join(Config.OUTPUT_DIR, 'detection'),
                            help='Output directory')
    detect_parser.add_argument('--model_path', default=None, help='Path to trained model weights')
    detect_parser.add_argument('--magnification', type=int, default=None, help='Image magnification')
    detect_parser.add_argument('--display', action='store_true', help='Display results')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract features from nuclei')
    extract_parser.add_argument('--input_image', required=True, help='Path to input image')
    extract_parser.add_argument('--output_dir', default=os.path.join(Config.OUTPUT_DIR, 'features'),
                             help='Output directory')
    extract_parser.add_argument('--model_path', default=None, help='Path to trained model weights')
    extract_parser.add_argument('--magnification', type=int, default=None, help='Image magnification')
    extract_parser.add_argument('--display', action='store_true', help='Display results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Run command
    if args.command == 'register':
        # Add timestamp to output directory
        args.output_dir = os.path.join(args.output_dir, f"registration_{timestamp}")
        register_images(args)
    elif args.command == 'process':
        # Add timestamp to output directory
        args.output_dir = os.path.join(args.output_dir, f"batch_{timestamp}")
        process_directory(args)
    elif args.command == 'detect':
        # Add timestamp to output directory
        args.output_dir = os.path.join(args.output_dir, f"detection_{timestamp}")
        detect_nuclei(args)
    elif args.command == 'extract':
        # Add timestamp to output directory
        args.output_dir = os.path.join(args.output_dir, f"features_{timestamp}")
        extract_features(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Run the pipeline
    run_pipeline()
