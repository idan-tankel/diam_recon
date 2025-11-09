# 3D Diamond Reconstruction

A Python-based tool for 3D diamond reconstruction using Radon transform on RGB images. This project implements custom Radon transform algorithms designed to process color-coded anatomical features of diamonds for tomographic reconstruction.

## Overview

This repository contains implementations for computing Radon projections on RGB images with specialized channel-specific processing. The primary use case is analyzing diamond anatomy where different features are color-coded (red, green, blue) and require different processing strategies during reconstruction.

## Key Features

- **RGB-aware Radon Transform**: Process each color channel with custom logic
- **Naive Implementation**: Custom pixel-by-pixel Radon transform with channel-specific masking
- **Scikit-image Integration**: Optimized implementation using scikit-image library
- **Streaming Support**: Process large image stacks without loading everything into memory
- **Batch Processing**: Efficient handling of multiple images
- **Gletz Angle Processing**: Special angle-dependent masking for green channel features

## Repository Structure

```
diam_recon/
├── src/
│   ├── __init__.py
│   └── radon_pipeline.py         # Main Radon transform implementations
├── data/
│   ├── diamond-anatomy.avif      # Reference diamond anatomy image
│   └── test_image.png            # Sample test image
├── tests/
│   └── test_streaming.py         # Unit tests for streaming functionality
├── test_quick.py                 # Quick validation test
├── test_rgb_demo.py              # RGB Radon transform demonstration
├── test_skimage_radon.py         # Comparison between implementations
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Build configuration
└── README.md                     # This file
```

### Output Files (Generated)

- `Figure_1.png` - Visualization outputs
- `radon_skimage_results.png` - Comparison results from scikit-image implementation
- `rgb_sinogram.png` - RGB sinogram visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/idan-tankel/diam_recon.git
cd diam_recon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy` - Array operations and numerical computing
- `scikit-image` - Radon transform implementation
- `imageio` - Image reading/writing
- `tqdm` - Progress bars
- `pytest` - Testing framework

## Usage

### Quick Test

Run a quick validation test:
```bash
python test_quick.py
```

### RGB Demo

Generate RGB Radon transform demonstration with visualization:
```bash
python test_rgb_demo.py
```

This will create a sinogram visualization saved as `rgb_sinogram.png`.

### Compare Implementations

Compare naive and scikit-image implementations:
```bash
python test_skimage_radon.py
```

This generates `radon_skimage_results.png` showing side-by-side comparison.

### Using the API

```python
from src.radon_pipeline import skimage_radon, naive_radon, _theta_from_angles
import numpy as np

# Load your RGB image (shape: H, W, 3)
rgb_image = ...  # Your image data

# Define projection angles
angles = _theta_from_angles(300)  # 300 angles from 0-360 degrees

# Compute Radon transform
sinogram = skimage_radon(rgb_image, angles, gletz_angle=30, gletz_threshold=40)
```

## Algorithm Details

### Color Channel Processing

The implementation uses specialized logic for each RGB channel:

- **Red Channel**: Pixels with high red values (>250) create black (0) entries in the sinogram
- **Green Channel**: High green values (>250) create gray (0.5) entries, but only at angles within a threshold of the "gletz angle" (default 30°)
- **Blue Channel**: Processed normally
- **White Pixels**: Pixels white in all channels (>240 threshold) remain white (1) in output

### Two Implementations

1. **`naive_radon()`**: Custom pixel-by-pixel implementation
   - Direct geometric projection
   - Explicit channel handling
   - Good for understanding the algorithm
   - Slower performance

2. **`skimage_radon()`**: Optimized scikit-image based
   - Leverages fast scikit-image Radon transform
   - Channel splitting and masking
   - Better performance for large images
   - Recommended for production use

### Output Format

Sinograms have shape `(n_angles, projection_length)` where:
- `n_angles` is the number of projection angles
- `projection_length = 2 * max_distance + 1` (max_distance from image center)
- Values range from 0 (black) to 1 (white), with 0.5 for gletz-masked features

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_streaming.py -v
```

## Technical Notes

- Images are processed in float32 format
- Default white pixel threshold: 240 (adjustable)
- Default red/green intensity threshold: 250 (adjustable)
- Gletz angle default: 30 degrees
- Gletz threshold default: 40 degrees
- All angles specified in degrees, converted to radians internally

## File Descriptions

### Source Files

- **`src/radon_pipeline.py`**: Main module containing:
  - `naive_radon()`: Custom Radon transform implementation
  - `skimage_radon()`: Scikit-image based implementation
  - `_theta_from_angles()`: Angle array generation
  - `_read_image_from_path()`: Image loading utility
  - `_compute_radon_for_image()`: Wrapper for single image processing

### Test Files

- **`test_quick.py`**: Quick sanity check with small RGB image
- **`test_rgb_demo.py`**: Full demonstration with 256x256 image, generates visualization
- **`test_skimage_radon.py`**: Comparison between implementations
- **`tests/test_streaming.py`**: Unit tests for batch/streaming processing

## Contributing

When adding features or making changes:
1. Ensure both implementations remain compatible
2. Update tests to cover new functionality
3. Run all tests before submitting changes
4. Update this README if adding new features or changing API

## License

[Add your license information here]

## Contact

[Add contact information here]
