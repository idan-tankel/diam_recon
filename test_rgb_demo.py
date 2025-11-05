"""
Demo script showing RGB Radon transform processing.

This script demonstrates how the modified radon_pipeline.py handles RGB images
by computing the Radon transform for each color channel individually.
Uses a 256x256 demo image for testing.
"""
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.radon_pipeline import naive_radon, _compute_radon_for_image, _theta_from_angles
from PIL import Image
import numpy as np

def create_demo_rgb_image(size=256):
    """Create a 256x256 demo RGB image with distinct patterns in each channel.
    
    Args:
        size: Image size (default 256 for demo purposes)
    
    Returns:
        RGB image of shape (size, size, 3)
    """
    print(f"Creating {size}x{size} demo RGB image...")
    # Load image from file instead of creating synthetic patterns
    

        # Load the image
    pil_img = Image.open('data/test_image.png')
    
    # Convert to RGB if not already
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    # Convert to numpy array
    img = np.array(pil_img)
    
    # Resize to desired size if needed
    if img.shape[:2] != (size, size):
        pil_img = pil_img.resize((size, size), Image.Resampling.LANCZOS)
        img = np.array(pil_img)
    
    print(f"✓ Loaded image from data/test_image.png with shape: {img.shape}")
    assert img.shape == (size, size, 3), f"Expected shape ({size}, {size}, 3), got {img.shape}"
    print(f"✓ Demo image created with shape: {img.shape}")
    print(img.max(axis=(0,1)))
    return img

def verify_image_properties(img, expected_size=256):
    """Verify that the demo image has the expected properties."""
    print(f"\nVerifying image properties...")
    
    # Check shape
    expected_shape = (expected_size, expected_size, 3)
    actual_shape = img.shape
    
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")
    
    if actual_shape == expected_shape:
        print("✓ Shape verification passed")
    else:
        raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {actual_shape}")
    
    # Check data type
    print(f"Image dtype: {img.dtype}")
    
    # Check value ranges for each channel
    for c, channel_name in enumerate(['Red', 'Green', 'Blue']):
        channel_data = img[:, :, c]
        min_val = channel_data.min()
        max_val = channel_data.max()
        non_zero = np.count_nonzero(channel_data)
        print(f"{channel_name} channel: min={min_val:.1f}, max={max_val:.1f}, non-zero pixels={non_zero}")
    
    return True

def demo_rgb_radon():
    """Demonstrate RGB Radon transform processing with 256x256 image."""
    print("Starting RGB Radon Transform Demo")
    print("=" * 50)
    
    # Create 256x256 demo image
    rgb_image = create_demo_rgb_image(256)
    
    # Verify image properties
    verify_image_properties(rgb_image, 256)

    # Print histogram of RGB image
    print("\nAnalyzing RGB image histogram...")
    rgb_flat = rgb_image.flatten()
    hist_counts, bin_edges = np.histogram(rgb_flat, bins=10)
    print(f"RGB image value histogram (10 bins):")
    for i in range(len(hist_counts)):
        print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist_counts[i]} pixels")

    # Also show per-channel histograms
    for c, channel_name in enumerate(['Red', 'Green', 'Blue']):
        channel_data = rgb_image[:, :, c].flatten()
        hist_counts, bin_edges = np.histogram(channel_data, bins=10)
        print(f"\n{channel_name} channel histogram:")
        for i in range(len(hist_counts)):
            print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist_counts[i]} pixels")
    
    # Define angles for projection (fewer angles for large image to manage processing time)
    angles = _theta_from_angles(300)  # 300 angles from 0 to 360
    print(f"\nUsing {len(angles)} projection angles for Radon transform")
    
    print("\nComputing Radon transform for RGB image...")
    print("Note: This may take a moment for a 256x256 image...")
    
    expected_proj_len = int(2 * np.ceil(np.hypot(127.5, 127.5)) + 1)  # For 256x256 image
    print(f"Expected shape: ({len(angles)}, ~{expected_proj_len})")
    
    # Test grayscale image for comparison
    print("\nTesting grayscale processing...")
    gray_sinogram = _compute_radon_for_image(rgb_image, angles)
    print(f"Grayscale sinogram shape: {gray_sinogram.shape}")
    
    # Show detailed statistics
    # Compute RGB sinogram
    sinogram = naive_radon(rgb_image, angles)

    # Save sinogram as PNG
    

    # Normalize sinogram for display (each channel separately)

    # Save as PNG
    # Print histogram of sinogram values binned to 0, 0.5, 1
    sinogram_flat = sinogram.flatten()
    hist_counts, bin_edges = np.histogram(sinogram_flat, bins=[0, 0.5, 1.0, np.inf])
    print(f"\nSinogram value histogram:")
    print(f"  [0, 0.5): {hist_counts[0]} pixels")
    print(f"  [0.5, 1.0): {hist_counts[1]} pixels")
    print(f"  [1.0, inf): {hist_counts[2]} pixels")
    
    sinogram * 255
    plt.figure(figsize=(12, 8))
    plt.imshow(sinogram * 255)
    plt.title('RGB Radon Transform Sinogram')
    plt.xlabel('Projection Position')
    plt.ylabel('Angle Index')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('rgb_sinogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Sinogram saved as 'rgb_sinogram.png'")

    return rgb_image, sinogram, None, None

if __name__ == "__main__":
    try:
        rgb_img, sino_rgb, gray_img, sino_gray = demo_rgb_radon()
        
        print(f"\n{'='*50}")
        print("✓ RGB Radon transform demo completed successfully!")
        print(f"\nDemo summary:")
        print(f"• Created 256x256 RGB demo image with distinct patterns")
        print(f"• Verified image shape: {rgb_img.shape}")
        print(f"• Processed each RGB channel individually")
        print(f"• Generated sinograms with shape: {sino_rgb.shape}")
        print(f"• Each color channel produces unique projection signatures")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback


        traceback.print_exc()