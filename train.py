"""
Training script for nuclei registration model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import logging
from tensorboardX import SummaryWriter

from config import Config
from data_preprocessing import StainNormalizer, create_dataloader
from model import NucleiRegistrationModel, EndToEndTrainableModel
from utils import calculate_metrics, save_checkpoint, load_checkpoint


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
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


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    writer=None
):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    running_loss = 0.0
    total_metrics = {metric: 0.0 for metric in Config.EVAL_METRICS}
    
    # Iterate over batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Get data
        fixed_img = batch['fixed'].to(device)
        moving_img = batch['moving'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        result = model(fixed_img, moving_img)
        
        # Calculate loss
        loss = 0.0
        
        # Similarity loss between fixed and transformed moving
        similarity_loss = criterion(result['transformed_moving'], fixed_img)
        loss += similarity_loss
        
        # Feature consistency loss (optional)
        if 'fixed_features' in result and 'moving_features' in result:
            feature_loss = nn.MSELoss()(
                result['fixed_features'],
                result['moving_features']
            )
            loss += 0.1 * feature_loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_metrics(
                fixed_img.detach().cpu().numpy(),
                result['transformed_moving'].detach().cpu().numpy(),
                Config.EVAL_METRICS
            )
            
            for metric, value in batch_metrics.items():
                total_metrics[metric] += value
        
        # Log to TensorBoard
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            
            # Log loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/similarity', similarity_loss.item(), global_step)
            
            if 'fixed_features' in result and 'moving_features' in result:
                writer.add_scalar('Loss/feature', feature_loss.item(), global_step)
            
            # Log metrics
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'Metrics/{metric}', value, global_step)
            
            # Log images (periodically)
            if batch_idx % 50 == 0:
                # Take first image in batch
                writer.add_image('Images/fixed', fixed_img[0], global_step)
                writer.add_image('Images/moving', moving_img[0], global_step)
                writer.add_image('Images/transformed', result['transformed_moving'][0], global_step)
    
    # Calculate average loss and metrics
    avg_loss = running_loss / len(dataloader)
    avg_metrics = {metric: value / len(dataloader) for metric, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    writer=None
):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        Average loss and metrics for validation
    """
    model.eval()
    
    running_loss = 0.0
    total_metrics = {metric: 0.0 for metric in Config.EVAL_METRICS}
    
    with torch.no_grad():
        # Iterate over batches
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation {epoch}")):
            # Get data
            fixed_img = batch['fixed'].to(device)
            moving_img = batch['moving'].to(device)
            
            # Forward pass
            result = model(fixed_img, moving_img)
            
            # Calculate loss
            loss = criterion(result['transformed_moving'], fixed_img)
            
            # Update running loss
            running_loss += loss.item()
            
            # Calculate metrics
            batch_metrics = calculate_metrics(
                fixed_img.cpu().numpy(),
                result['transformed_moving'].cpu().numpy(),
                Config.EVAL_METRICS
            )
            
            for metric, value in batch_metrics.items():
                total_metrics[metric] += value
    
    # Calculate average loss and metrics
    avg_loss = running_loss / len(dataloader)
    avg_metrics = {metric: value / len(dataloader) for metric, value in total_metrics.items()}
    
    # Log to TensorBoard
    if writer is not None:
        # Log loss
        writer.add_scalar('Loss/val', avg_loss, epoch)
        
        # Log metrics
        for metric, value in avg_metrics.items():
            writer.add_scalar(f'Metrics/{metric}_val', value, epoch)
        
        # Log sample images
        if len(dataloader) > 0:
            # Get a sample batch
            sample_batch = next(iter(dataloader))
            fixed_img = sample_batch['fixed'].to(device)
            moving_img = sample_batch['moving'].to(device)
            
            # Forward pass
            with torch.no_grad():
                result = model(fixed_img, moving_img)
            
            # Log images
            writer.add_image('Images_val/fixed', fixed_img[0], epoch)
            writer.add_image('Images_val/moving', moving_img[0], epoch)
            writer.add_image('Images_val/transformed', result['transformed_moving'][0], epoch)
    
    return avg_loss, avg_metrics


def train_model():
    """Train the nuclei registration model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train nuclei registration model')
    parser.add_argument('--data_root', type=str, default=Config.DATA_ROOT, help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=Config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging and TensorBoard
    logger = setup_logging(output_dir)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Log configuration
    logger.info(f"Training configuration:")
    logger.info(f"  Data root: {args.data_root}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Initialize stain normalizer
    normalizer = StainNormalizer(method=Config.STAIN_NORM_METHOD)
    if os.path.exists(Config.REFERENCE_STAIN_PATH):
        normalizer.set_reference(Config.REFERENCE_STAIN_PATH)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    
    # Training data
    train_dir = os.path.join(args.data_root, 'train')
    if not os.path.exists(train_dir):
        logger.warning(f"Training directory not found: {train_dir}. Using data_root instead.")
        train_dir = args.data_root
    
    train_loader = create_dataloader(
        train_dir,
        batch_size=args.batch_size,
        is_training=True,
        normalizer=normalizer
    )
    
    # Validation data
    val_dir = os.path.join(args.data_root, 'val')
    if os.path.exists(val_dir):
        val_loader = create_dataloader(
            val_dir,
            batch_size=args.batch_size,
            is_training=False,
            normalizer=normalizer
        )
    else:
        logger.warning(f"Validation directory not found: {val_dir}. Using a subset of training data for validation.")
        
        # Split training data for validation
        train_size = int(0.8 * len(train_loader.dataset))
        val_size = len(train_loader.dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_loader.dataset,
            [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = EndToEndTrainableModel(
        detector_weights_path=args.pretrained_path if args.use_pretrained else None,
        device=device
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize criterion
    criterion = nn.MSELoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = Config.PATIENCE
    patience_counter = 0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        # Train one epoch
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            writer
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            writer
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log results
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Train Metrics: {train_metrics}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Metrics: {val_metrics}")
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(output_dir, 'best_model.pth')
            )
            logger.info(f"  Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.num_epochs - 1,
        val_loss,
        os.path.join(output_dir, 'final_model.pth')
    )
    
    logger.info("Training completed!")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, output_dir


if __name__ == "__main__":
    # Train the model
    model, output_dir = train_model()
