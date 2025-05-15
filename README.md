
# Nuclei Points-Based Registration Model

A universal multi-scale nuclei registration framework for histological images at various magnifications and staining methods.

## Overview

This project implements a robust nuclei-based registration approach for aligning histological images. It works across:
- Different magnifications (5x, 10x, 20x, 40x)
- Various tissue types
- Different staining methods (H&E, immunohistochemistry, etc.)

The model uses detected nuclei as landmarks for registration, extracting rich features that enable accurate alignment even in challenging cases.

## Features

- **Multi-scale architecture**: Works with different magnification levels
- **Stain normalization**: Handles various staining methods and intensities
- **Robust feature extraction**: Combines geometric, intensity, texture, and neighborhood features
- **Multiple registration methods**: ICP, RANSAC, and CPD algorithms
- **End-to-end pipeline**: From preprocessing to visualization of results

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nuclei-registration.git
cd nuclei-registration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- NumPy
- SciPy
- Matplotlib
- scikit-image
- TensorboardX
- tqdm
- medpy

## Usage

### Basic Registration

To register two histological images:

```bash
python main.py register --fixed_image path/to/fixed_image.png --moving_image path/to/moving_image.png --output_dir results/registration
```

### Batch Processing

Process an entire directory of image pairs:

```bash
python main.py process --input_dir path/to/image_pairs --output_dir results/batch
```

### Nuclei Detection

Detect nuclei in a single image:

```bash
python main.py detect --input_image path/to/image.png --output_dir results/detection
```

### Feature Extraction

Extract features from detected nuclei:

```bash
python main.py extract --input_image path/to/image.png --output_dir results/features
```

### Training the Model

To train the registration model:

```bash
python train.py --data_root path/to/training_data --output_dir results/training
```

### Evaluation

Evaluate the model on a test dataset:

```bash
python evaluate.py --data_dir path/to/test_data --output_dir results/evaluation
```

## Project Structure

- `config.py`: Configuration settings
- `data_preprocessing.py`: Stain normalization and data loading
- `nuclei_detection.py`: Nuclei detection models and pipeline
- `feature_extraction.py`: Feature extraction from nuclei
- `registration.py`: Registration algorithms (ICP, RANSAC, CPD)
- `model.py`: Main model integrating all components
- `train.py`: Training pipeline
- `evaluate.py`: Evaluation metrics and visualizations
- `utils.py`: Utility functions
- `main.py`: Command-line interface

## Advanced Configuration

Modify `config.py` to adjust parameters such as:

- Nuclei detection thresholds
- Feature extraction settings
- Registration parameters
- Training hyperparameters

## Example Results

When registering histological images, the system provides visualizations including:
- Original fixed and moving images
- Transformed moving image
- Difference map
- Overlay visualization
- Checkerboard pattern
- Detected nuclei points

## Citing This Work

If you use this code in your research, please cite our work:

```
@article{nuclei_registration_2025,
  title={Universal Multi-Scale Nuclei-Based Registration for Histological Images},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

We thank the contributors of the following open-source projects:
- PyTorch
- OpenCV
- scikit-image
- MedPy
