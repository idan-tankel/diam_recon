"""Radon transform pipeline with streaming and batching.

This module provides functions to compute Radon projections for large stacks
without loading all images into memory at once. It supports:
- directory input (images)
- .npy stacks (memory-mapped)
- batched processing with a numpy.memmap output to write results incrementally

Design notes:
- Output shape is (N, n_angles, proj_len) and dtype float32.
- For directories, images are processed in filename-sorted order.
"""
from typing import Iterable, Union, Tuple
import os
from pathlib import Path
import numpy as np
import imageio
import math
try:
    from skimage.transform import radon
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def _theta_from_angles(angles: Union[int, Iterable[float]]) -> np.ndarray:
    if isinstance(angles, int):
        # angles number of samples from 0 to 360 degrees
        return np.linspace(0., 360., angles, endpoint=False)
    return np.asarray(list(angles), dtype=float)


def _read_image_from_path(path: str) -> np.ndarray:
    img = imageio.v2.imread(path)
    arr = np.asarray(img)
    # Keep RGB channels for individual processing
    if arr.ndim == 3 and arr.shape[2] == 3:
        # RGB image - keep all channels
        return arr.astype(np.float32)
    elif arr.ndim == 2:
        # Grayscale image - return as is
        return arr.astype(np.float32)
    else:
        # For other cases (RGBA, etc.), take first 3 channels or convert appropriately
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3].astype(np.float32)
        else:
            # fallback to grayscale
            return arr[..., 0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)

def _compute_radon_for_image(img: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute Radon transform for an image, handling both grayscale and RGB.
    
    Args:
        img: 2D grayscale (H, W) or 3D RGB (H, W, 3) image
        theta: 1D array of angles
        
    Returns:
        For grayscale: (len(theta), proj_len)
        For RGB: (len(theta), proj_len, 3)
    """
    sino = naive_radon(img, theta)
    
    if sino.ndim == 2:
        # Grayscale: transpose from (proj_len, len(theta)) to (len(theta), proj_len)
        return sino.T.astype(np.float32)
    else:
        # RGB: transpose from (proj_len, len(theta), 3) to (len(theta), proj_len, 3)
        return sino.transpose(1, 0, 2).astype(np.float32)


def naive_radon(img: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Naive Radon transform implementation supporting RGB channels.

    Args:
        img: 2D image (H, W) for grayscale or 3D image (H, W, C) for RGB
        theta: 1D array of angles in degrees

    Returns:
        sino: ndarray 
            - For grayscale: shape (proj_len, len(theta))
            - For RGB: shape (proj_len, len(theta), 3) with each channel processed individually
    """
    H, W, C = img.shape
    assert C == 3, "RGB image must have 3 channels"
        
        # Get projection parameters from any channel (they're the same)
    max_dist = int(math.ceil(np.hypot((W - 1) / 2.0, (H - 1) / 2.0)))
    proj_len = 2 * max_dist + 1
        
        # Initialize output which starts with zeros (black background)
    sinogram = np.zeros((proj_len, len(theta)), dtype=np.float32)
    # center coordinates
    # assuming the rotation center is at the image center
    a_x = (W - 1) / 2.0
    a_y = (H - 1) / 2.0

    # pixel coordinates relative to center
    ys = np.arange(H) - a_y
    xs = np.arange(W) - a_x
    X, Y = np.meshgrid(xs, ys)

    # max distance from center determines projection length
    max_dist = int(math.ceil(np.hypot(a_x, a_y)))
    # detector bins from -max_dist .. +max_dist inclusive
    # bins form 0 to 2*max_dist
    s_bins = np.arange(0, 2 * max_dist + 1)
    proj_len = s_bins.size

    thetas = np.deg2rad(theta)
    sinogram = np.ones((thetas.size, proj_len), dtype=np.float32)

    # For each angle, compute projection by binning pixel contributions
    for j, t in enumerate(thetas):
        cosine = math.cos(t)
        sine = math.sin(t)
        
        # Iterate through each pixel in the image directly (no ravel)
        for i in range(H):
            for k in range(W):
                # Get pixel coordinate relative to center
                x_coord = X[i, k]  # k - cx
                y_coord = Y[i, k]  # i - cy
                a_s = max_dist  # offset is in the middle since the number of bins 
                # Get pixel value
                
                # Compute projection coordinate for this pixel: x*cos + y*sin
                # note that the original formula uses (a_s - s * x_coord + c * y_coord)
                proj_coord = a_s - sine * x_coord + cosine * y_coord
                
                # Map coordinate to bin index
                bin_idx = int(np.round(proj_coord))
                
                # Accumulate pixel value into appropriate bin
                if 0 <= bin_idx < proj_len:
                    # For RGB images, we need to handle each channel separately
                    # This is within the single-channel _naive_radon_2d function,
                    # so we just accumulate the single pixel value as before
                    # Skip white pixels (assuming white is close to [255, 255, 255] or high values)
                    pixel_rgb = img[i, k, :]
                    if np.all(pixel_rgb >= 240):  # Adjust threshold as needed
                        continue
                    
                    for channel_index in range(3):
                        pixel_value = img[i, k, channel_index]
                        # that is red channel logic
                        intensity_threshold = 250
                        if channel_index == 0:
                            # check if the pixel value is 255,0,0
                            if pixel_value > intensity_threshold:
                                # print("updated red channel")
                                sinogram[j, bin_idx] = 0 # blacken the bin
                            else:
                                pass
                        # that is green channel logic
                        elif channel_index == 1:

                            if pixel_value > intensity_threshold:
                                # For that, we should check if the angle of the rotation is close
                                # enough to the gletz angle as given as an argument
                                gletz_angle = 30
                                Threshold = 40 # degrees
                                gletz_angle = np.deg2rad(gletz_angle)
                                Threshold = np.deg2rad(Threshold)
                                if abs(theta[j] - gletz_angle) < Threshold:
                                    # the pixel should be greyed
                                    sinogram[j, bin_idx] = 0.5
                                else:
                                    pass
                        # that is blue channel logic
                        elif channel_index == 2:
                            # as the blue pixel is white, we do nothing
                            pass
    
    return sinogram


def skimage_radon(img: np.ndarray, theta: np.ndarray, gletz_angle: float = 30, gletz_threshold: float = 40, white_intensity_threshold: float = 240) -> np.ndarray:
    """Radon transform using scikit-image implementation for RGB channels.
    
    Args:
        img: 3D RGB image (H, W, 3)
        theta: 1D array of angles in degrees
        gletz_angle: Special angle in degrees for green channel masking
        gletz_threshold: Threshold in degrees for green channel angle masking
        white_intensity_threshold: Intensity threshold to consider a pixel as white in all channels
        
    Returns:
        sino: ndarray shape (len(theta), proj_len) - composite sinogram
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is not available. Please install it with: pip install scikit-image")
    
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")
       
    H, W, C = img.shape
    # in order to handle white pixels correctly, we should keep them as is (1 in the output)
    # for that, we will convert them to zero (which after the radon will become zero again)
    # and than we will convert them back to one after the radon.
    # notice that this change should be done only if the values in all channels are high enough to be considered white.
    # Apply white pixel threshold from function argument
    white_mask = np.all(img >= white_intensity_threshold, axis=2)
    if np.any(white_mask):
        img_processed = img.copy()
        img_processed[white_mask] = 0  # set white pixels to black for processing
    else:
        img_processed = img.copy()
    # continue work on the img_processed


    # Split RGB image into individual channels
    red_channel = img_processed[:, :, 0]
    green_channel = img_processed[:, :, 1]
    # we should catch things that are likely to be red.
    red_intensity_threshold = 250
    # 1. Red channel logic: red values in the initial image would be zero
    red_mask = red_channel > red_intensity_threshold
    # notice that since that is the processed image, values above the threshold are red and not white
    # however, the red mask should be the same as the original mask
    if np.any(red_mask):
        # Find projections of bright red pixels and set those sinogram values to 0
        red_sino_thresholded = radon(red_mask.astype(np.float32), theta=theta, circle=False)
        # Initialize composite_sino with ones (white background) using the shape from red sinogram
        composite_sino = np.ones_like(red_sino_thresholded, dtype=np.float32)
        # since the red_sino_threshold is opposite - red pixels would correspond to >0 values in the sinogram (not necessarily one)
        composite_sino[red_sino_thresholded > 0] = 0
    else:
        # If no red pixels, create a dummy radon to get the right shape
        dummy_red = radon(np.zeros_like(red_channel), theta=theta, circle=False)
        composite_sino = np.ones_like(dummy_red, dtype=np.float32)
    
    # 3. Green channel logic: values other than one would be half (according to gletz mask)
    green_mask = green_channel > red_intensity_threshold
    if np.any(green_mask):
        green_sino_thresholded = radon(green_mask.astype(np.float32), theta=theta, circle=False)
        # Apply gletz angle specific effects
        # Find angles within gletz threshold range
        angle_mask = np.abs(theta - gletz_angle) < gletz_threshold
        angle_indices = np.where(angle_mask)[0]
        
        # Apply gletz effect only for qualifying angles and green pixels
        for j in angle_indices:
            composite_sino[j, green_sino_thresholded[j, :] > 0] = 0.5
    
    print(f"✓ Scikit-image radon transform completed. Output shape: {composite_sino.shape}")
    
    return composite_sino.astype(np.float32)





    