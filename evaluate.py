"""
Evaluation script for nuclei registration model.
"""

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from glob import glob

from config import Config
from model import NucleiRegistrationModel
from utils import calculate_metrics, create_evaluation_visualizations


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'evaluation.log')
    
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


def evaluate_on_dataset(
    model,
    data_dir,
    output_dir,
    magnification=None,
    metrics=None,
    save_visualizations=True
):
    """
    Evaluate model on a dataset of image pairs.
    
    Args:
        model: The registration model to evaluate
        data_dir: Directory containing image pairs
        output_dir: Directory to save results
        magnification: Magnification level of images
        metrics: List of metrics to calculate
        save_visualizations: Whether to save visualization images
        
    Returns:
        Dictionary of evaluation results
    """
    # Create logger
    logger = setup_logging(output_dir)
    
    # Create output directories
    results_dir = os.path.join(output_dir, 'results')
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    
    if save_visualizations:
        os.makedirs(visualizations_dir, exist_ok=True)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Set metrics
    if metrics is None:
        metrics = Config.EVAL_METRICS
    
    # Find image pairs
    image_pairs = []
    
    # Try to find image pairs with specific naming conventions
    fixed_images = sorted(glob(os.path.join(data_dir, '*_fixed.*')))
    moving_images = sorted(glob(os.path.join(data_dir, '*_moving.*')))
    
    if len(fixed_images) == len(moving_images) and len(fixed_images) > 0:
        # Paired by naming convention
        image_pairs = list(zip(fixed_images, moving_images))
    else:
        # Try to find pairs organized in subdirectories
        pair_dirs = sorted(glob(os.path.join(data_dir, 'pair*')))
        
        if len(pair_dirs) > 0:
            for pair_dir in pair_dirs:
                fixed_image = glob(os.path.join(pair_dir, '*_fixed.*'))
                moving_image = glob(os.path.join(pair_dir, '*_moving.*'))
                
                if len(fixed_image) == 1 and len(moving_image) == 1:
                    image_pairs.append((fixed_image[0], moving_image[0]))
        else:
            # Just pair sequential images as a fallback
            all_images = sorted(glob(os.path.join(data_dir, '*.png')) + 
                              glob(os.path.join(data_dir, '*.jpg')) + 
                              glob(os.path.join(data_dir, '*.tif')))
            
            if len(all_images) % 2 == 0:
                for i in range(0, len(all_images), 2):
                    image_pairs.append((all_images[i], all_images[i+1]))
    
    # Check if we found pairs
    if len(image_pairs) == 0:
        logger.error(f"No image pairs found in {data_dir}")
        return None
    
    logger.info(f"Found {len(image_pairs)} image pairs for evaluation")
    
    # Initialize results storage
    results = {
        'pairs': [],
        'summary': {metric: {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')} for metric in metrics}
    }
    
    # Process each pair
    for idx, (fixed_path, moving_path) in enumerate(tqdm(image_pairs, desc="Evaluating")):
        # Load images
        fixed_image = cv2.imread(fixed_path)
        fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB)
        
        moving_image = cv2.imread(moving_path)
        moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)
        
        # Register images
        try:
            registration_result = model.register_images(
                fixed_image, 
                moving_image,
                fixed_magnification=magnification,
                moving_magnification=magnification
            )
        except Exception as e:
            logger.error(f"Error processing pair {idx} ({fixed_path}, {moving_path}): {e}")
            continue
        
        # Extract transformed image
        transformed_moving = registration_result['transformed_moving']
        
        # Calculate metrics
        pair_metrics = calculate_metrics(fixed_image, transformed_moving, metrics)
        
        # Save results for this pair
        pair_result = {
            'fixed_path': fixed_path,
            'moving_path': moving_path,
            'metrics': pair_metrics,
            'transformation': registration_result['registration_result']['transformation'].tolist() 
                              if 'transformation' in registration_result['registration_result'] else None
        }
        
        results['pairs'].append(pair_result)
        
        # Update summary stats
        for metric, value in pair_metrics.items():
            results['summary'][metric]['min'] = min(results['summary'][metric]['min'], value)
            results['summary'][metric]['max'] = max(results['summary'][metric]['max'], value)
            results['summary'][metric]['mean'] += value / len(image_pairs)
        
        # Save visualizations
        if save_visualizations:
            # Create visualizations
            vis_images = create_evaluation_visualizations(
                fixed_image,
                moving_image,
                transformed_moving,
                registration_result
            )
            
            # Save visualization images
            for vis_name, vis_image in vis_images.items():
                output_path = os.path.join(
                    visualizations_dir,
                    f"pair_{idx:04d}_{vis_name}.png"
                )
                
                # Convert to BGR for OpenCV
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Calculate standard deviations
    for metric in metrics:
        mean_value = results['summary'][metric]['mean']
        sum_squared_diff = sum((pair['metrics'][metric] - mean_value) ** 2 for pair in results['pairs'])
        results['summary'][metric]['std'] = np.sqrt(sum_squared_diff / len(results['pairs']))
    
    # Log summary results
    logger.info("Evaluation summary:")
    for metric, stats in results['summary'].items():
        logger.info(f"  {metric}:")
        logger.info(f"    Mean: {stats['mean']:.4f}")
        logger.info(f"    Std: {stats['std']:.4f}")
        logger.info(f"    Min: {stats['min']:.4f}")
        logger.info(f"    Max: {stats['max']:.4f}")
    
    # Save results to JSON
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {results_path}")
    
    # Create summary visualization
    if save_visualizations:
        # Select a few representative pairs
        if len(results['pairs']) > 3:
            # Get pair with median metric value
            median_idx = np.argsort([p['metrics'][metrics[0]] for p in results['pairs']])[len(results['pairs']) // 2]
            
            # Get pair with best metric value
            best_idx = np.argmin([p['metrics'][metrics[0]] for p in results['pairs']])
            
            # Get pair with worst metric value
            worst_idx = np.argmax([p['metrics'][metrics[0]] for p in results['pairs']])
            
            sample_indices = [best_idx, median_idx, worst_idx]
        else:
            sample_indices = range(len(results['pairs']))
        
        # Create figure with sample pairs
        plt.figure(figsize=(15, 5 * len(sample_indices)))
        
        for i, idx in enumerate(sample_indices):
            pair_info = results['pairs'][idx]
            
            # Load images
            fixed_path = pair_info['fixed_path']
            moving_path = pair_info['moving_path']
            
            fixed_image = cv2.imread(fixed_path)
            fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB)
            
            moving_image = cv2.imread(moving_path)
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)
            
            # Register images
            registration_result = model.register_images(
                fixed_image, 
                moving_image,
                fixed_magnification=magnification,
                moving_magnification=magnification
            )
            
            transformed_moving = registration_result['transformed_moving']
            
            # Plot images
            plt.subplot(len(sample_indices), 3, i * 3 + 1)
            plt.imshow(fixed_image)
            plt.title(f"Fixed Image")
            if i == 0:
                plt.ylabel("Best Registration")
            elif i == 1:
                plt.ylabel("Median Registration")
            elif i == 2:
                plt.ylabel("Worst Registration")
            
            plt.subplot(len(sample_indices), 3, i * 3 + 2)
            plt.imshow(moving_image)
            plt.title(f"Moving Image")
            
            plt.subplot(len(sample_indices), 3, i * 3 + 3)
            plt.imshow(transformed_moving)
            plt.title(f"Transformed Moving\n{metrics[0]}: {pair_info['metrics'][metrics[0]]:.4f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, 'summary_visualization.png'), dpi=200)
    
    return results


def evaluate_model():
    """Evaluate the nuclei registration model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate nuclei registration model')
    parser.add_argument('--data_dir', type=str, default=os.path.join(Config.DATA_ROOT, 'test'),
                        help='Directory containing test image pairs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=os.path.join(Config.OUTPUT_DIR, 'evaluation'),
                        help='Output directory for evaluation results')
    parser.add_argument('--magnification', type=int, default=None,
                        help='Magnification level of images')
    parser.add_argument('--no_visualizations', action='store_true',
                        help='Disable saving visualization images')
    parser.add_argument('--registration_method', type=str, default=Config.REGISTRATION_METHOD,
                        choices=['icp', 'ransac', 'cpd'],
                        help='Registration method to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = NucleiRegistrationModel(
        detector_weights_path=args.model_path,
        registration_method=args.registration_method
    )
    
    # Evaluate on dataset
    results = evaluate_on_dataset(
        model,
        args.data_dir,
        args.output_dir,
        magnification=args.magnification,
        save_visualizations=not args.no_visualizations
    )
    
    return results


if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_model()
