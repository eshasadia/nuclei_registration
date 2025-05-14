"""
Configuration settings for the nuclei registration model.
"""

import os
import torch

class Config:
    # General settings
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "results"
    
    # Dataset settings
    DATA_ROOT = "data/hyreco"
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1  # Remaining 0.1 is for test
    
    # Preprocessing settings
    STAIN_NORM_METHOD = "macenko"  # Options: "macenko", "reinhard", "vahadane"
    REFERENCE_STAIN_PATH = "data/reference_h_e.png"
    TARGET_SIZE = (512, 512)
    
    # Nuclei detection settings
    MODEL_TYPE = "unet"  # Options: "unet", "mask_rcnn", "cellpose"
    MAGNIFICATION_LEVELS = [5, 10, 20, 40]
    INTENSITY_THRESHOLD = 0.5
    NUCLEI_MIN_SIZE = 10  # Minimum nucleus size in pixels
    
    # Feature extraction settings
    FEATURE_TYPES = ["geometric", "intensity", "texture"]
    PATCH_SIZE = 32  # Size of patch around nuclei for feature extraction
    
    # Registration settings
    REGISTRATION_METHOD = "cpd"  # Options: "cpd", "icp", "ransac"
    MAX_ITERATIONS = 100
    CONVERGENCE_THRESHOLD = 1e-6
    REGULARIZATION_WEIGHT = 2.0
    OUTLIER_WEIGHT = 0.1
    
    # Training settings
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 10  # For early stopping
    
    # Evaluation settings
    EVAL_METRICS = ["mse", "dice", "hausdorff", "tre"]
    SAVE_VISUALIZATIONS = True
    
    # Create necessary directories
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "visualizations"), exist_ok=True)

# Create directories when the module is imported
Config.create_dirs()
