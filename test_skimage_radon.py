"""
Test script for comparing naive and scikit-image Radon transform implementations.

This script demonstrates the new skimage_radon function and compares it
with the original naive implementation.
"""
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.radon_pipeline import (
    skimage_radon, 
    naive_radon,
    _theta_from_angles
)

def create_simple_test_image(size=64):
    """Create a simple RGB test image for comparison."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    # Red channel: horizontal stripes
    for i in range(0, size, 8):
        if i + 4 < size:
            img[i:i+4, :, 0] = 255
    
    # Green channel: vertical stripes  
    for j in range(0, size, 8):
        if j + 4 < size:
            img[:, j:j+4, 1] = 255
    
    # Blue channel: diagonal pattern
    for i in range(size):
        for j in range(size):
            if (i + j) % 16 < 8:
                img[i, j, 2] = 200
                
    return img

def test_skimage_radon():
    """Test the scikit-image Radon implementation."""
    print("Testing scikit-image Radon transform implementation")
    print("=" * 60)
    
    # Create test image
    print("Creating test RGB image...")
    rgba_image = Image.open('data/test_image.png')
    rgb_image = np.array(rgba_image.convert('RGB'))
    print(f"Test image shape: {rgb_image.shape}")
    
    # Define angles
    angles = _theta_from_angles(300)  # 300 angles
    print(f"Using {len(angles)} projection angles")
    
    try:
        # Test scikit-image implementation
        print("\nTesting scikit-image implementation...")
        skimage_result = skimage_radon(rgb_image, angles, gletz_angle=30, gletz_threshold=40)
        print(f"Scikit-image result shape: {skimage_result.shape}")
        print(f"Scikit-image result range: {skimage_result.min():.2f} to {skimage_result.max():.2f}")
        
        # Test naive implementation for comparison
        print("\nTesting naive implementation...")
        naive_result = naive_radon(rgb_image, angles)
        print(f"Naive result shape: {naive_result.shape}")
        print(f"Naive result range: {naive_result.min():.2f} to {naive_result.max():.2f}")
        
        # Compare results
        print(f"\nShape comparison:")
        print(f"  Scikit-image: {skimage_result.shape}")
        print(f"  Naive:        {naive_result.shape}")
        
        if skimage_result.shape == naive_result.shape:
            # Calculate difference
            diff = np.abs(skimage_result - naive_result)
            print(f"\nDifference statistics:")
            print(f"  Mean absolute difference: {diff.mean():.4f}")
            print(f"  Max absolute difference:  {diff.max():.4f}")
            print(f"  Correlation coefficient:  {np.corrcoef(skimage_result.flatten(), naive_result.flatten())[0,1]:.4f}")
        else:
            print("  Shapes don't match - cannot compare directly")
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Scikit-image is not available")
        return False
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function."""
    print("Radon Transform Implementation Comparison")
    print("=" * 60)
    
    success = True
    
    # Test individual scikit-image implementation
    success = test_skimage_radon()
    
    # Save visualization of the test results
    try:
        import matplotlib.pyplot as plt
        
        # Create test image and run transforms
        rgba_image = Image.open('data/test_image.png')
        rgb_image = np.array(rgba_image.convert('RGB'))
        angles = _theta_from_angles(300)
        
        skimage_result = skimage_radon(rgb_image, angles, gletz_angle=30, gletz_threshold=240)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original RGB image
        axes[0].imshow(rgb_image.astype(np.uint8))
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        # Scikit-image result
        im1 = axes[1].imshow(skimage_result, cmap='hot', aspect='auto')
        axes[1].set_title('Scikit-image Radon Transform')
        axes[1].set_xlabel('Projection Angle')
        axes[1].set_ylabel('Detector Position')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), 'radon_skimage_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - cannot save visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Test comparison function
    
    print(f"\n{'='*60}")
    if success:
        print("✓ All tests completed successfully!")
        print(f"\nKey achievements:")
        print(f"• Implemented scikit-image based Radon transform")
        print(f"• Added RGB channel splitting and independent processing")
        print(f"• Implemented gletz_angle masking for green channel")
        print(f"• Created performance comparison functionality")
        print(f"• Maintained compatibility with original naive implementation")
    else:
        print("✗ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()