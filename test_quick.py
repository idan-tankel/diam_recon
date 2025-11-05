import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from src.radon_pipeline import naive_radon, _theta_from_angles
import numpy as np

# Test with a simple RGB image
test_img = np.zeros((64, 64, 3), dtype=np.float32)
test_img[20:40, 20:40, 0] = 255  # Red square
test_img[30:50, 30:50, 1] = 255  # Green square
angles = _theta_from_angles(10)  # Just 10 angles for quick test

print('Testing naive_radon...')
try:
    result = naive_radon(test_img, angles)
    print(f'Success! Result shape: {result.shape}')
    print(f'Result min: {result.min()}, max: {result.max()}, mean: {result.mean():.3f}')
    # Check if we have variety in values (not all the same)
    unique_vals = np.unique(result)
    print(f'Number of unique values: {len(unique_vals)}')
    print(f'Sample unique values: {unique_vals[:10]}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()