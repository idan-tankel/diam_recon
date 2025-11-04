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


def _get_image_paths(input_path: str) -> Tuple[str, int]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == '.npy':
        return 'npy', 0
    if p.is_dir():
        files = sorted([str(p / f) for f in os.listdir(p)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        return files, len(files)
    raise ValueError(f"Input {input_path} is neither a directory of images nor a .npy file")


def _theta_from_angles(angles: Union[int, Iterable[float]]) -> np.ndarray:
    if isinstance(angles, int):
        return np.linspace(0., 180., angles, endpoint=False)
    return np.asarray(list(angles), dtype=float)


def _read_image_from_path(path: str) -> np.ndarray:
    img = imageio.v2.imread(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        # convert to grayscale by taking first channel (fast) - user can supply grayscale stacks
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _read_image_from_npy(arr: np.ndarray, idx: int) -> np.ndarray:
    return np.asarray(arr[idx]).astype(np.float32)


def _compute_radon_for_image(img: np.ndarray, theta: np.ndarray, circle: bool) -> np.ndarray:
    # Use a simple, explicit (but slow) Radon implementation for demonstration.
    # This computes the projection by summing pixel values along lines at each angle.
    # It returns an array with shape (len(theta), proj_len) to match the previous API.
    # naive_radon no longer supports masking; ignore `circle` here.
    sino = naive_radon(img, theta)
    return sino.T.astype(np.float32)  # (len(theta), proj_len)


def naive_radon(img: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Naive Radon transform implementation (no masking).

    Args:
        img: 2D image (H, W)
        theta: 1D array of angles in degrees

    Returns:
        sino: ndarray shape (proj_len, len(theta)) where proj_len is number of bins along detector
    """
    H, W = img.shape
    # center coordinates
    # assuming the rotation center is at the image center
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # pixel coordinates relative to center
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    X, Y = np.meshgrid(xs, ys)

    # max distance from center determines projection length
    max_dist = int(math.ceil(np.hypot(cx, cy)))
    # detector bins from -max_dist .. +max_dist inclusive
    s_bins = np.arange(-max_dist, max_dist + 1)
    proj_len = s_bins.size

    thetas = np.deg2rad(theta)
    sino = np.zeros((proj_len, thetas.size), dtype=np.float32)

    # flatten arrays for iteration
    Xf = X.ravel()
    Yf = Y.ravel()
    Vf = img.ravel().astype(np.float32)
    Mf = mask.ravel()

    # For each angle, compute projection by binning pixel contributions
    for j, t in enumerate(thetas):
        c = math.cos(t)
        s = math.sin(t)
        # compute projection coordinate for each pixel: x*cos + y*sin
        coords = Xf * c + Yf * s
    # include all pixels (mask is all-True)
    coords = coords[Mf]
    vals = Vf[Mf]
        # map coords to bin indices
        idx = np.round(coords - s_bins[0]).astype(int)
        # accumulate values into bins manually
        for k, v in zip(idx, vals):
            if 0 <= k < proj_len:
                sino[k, j] += v

    return sino


def stream_radon(input_path: str,
                 output_path: str,
                 angles: Union[int, Iterable[float]] = 180,
                 batch_size: int = 32,
                 per_image: bool = False,
                 circle: bool = True) -> np.ndarray:
    """Process the input stack in batches and write projections to disk.

    Args:
        input_path: directory with images or path to .npy file (3D array N,H,W)
        output_path: path to output .npy file (memmap) or directory when per_image=True
        angles: either int (number of evenly spaced angles between 0..180) or iterable of angles
        batch_size: number of images to process per batch
        per_image: if True, save one .npy per input image into output_path (which must be a directory)
        circle: passed to skimage.radon

    Returns:
        If per_image is False, returns the path to the output .npy file.
        If per_image is True, returns None.
    """
    theta = _theta_from_angles(angles)

    items, count = _get_image_paths(input_path)

    # prepare input iterator
    if items == 'npy':
        arr = np.load(input_path, mmap_mode='r')
        if arr.ndim != 3:
            raise ValueError('.npy file must be a 3D array (N,H,W)')
        N = arr.shape[0]
        def reader(i):
            return _read_image_from_npy(arr, i)
        indices = range(N)
    else:
        files = items
        N = len(files)
        def reader(i):
            return _read_image_from_path(files[i])
        indices = range(N)

    if N == 0:
        raise ValueError('No images found to process')

    # determine projection shape using first image
    first_img = reader(0)
    first_sino = _compute_radon_for_image(first_img, theta, circle)
    n_angles = first_sino.shape[0]
    proj_len = first_sino.shape[1]

    if per_image:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save first
        np.save(str(out_dir / f'proj_0.npy'), first_sino)
        start = 1
        for i in range(start, N):
            img = reader(i)
            sino = _compute_radon_for_image(img, theta, circle)
            np.save(str(out_dir / f'proj_{i}.npy'), sino)
        return None

    # create memmap output file
    out_shape = (N, n_angles, proj_len)
    out_dtype = np.float32
    # ensure parent dir exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # create memmap file
    mm = np.lib.format.open_memmap(str(out_path), mode='w+', dtype=out_dtype, shape=out_shape)

    # write first
    mm[0] = first_sino

    # process in batches
    for batch_start in range(1, N, batch_size):
        batch_end = min(N, batch_start + batch_size)
        for i in range(batch_start, batch_end):
            img = reader(i)
            sino = _compute_radon_for_image(img, theta, circle)
            mm[i] = sino

    # flush to disk
    del mm
    return str(out_path)


def compute_radon_stack(stack: np.ndarray, angles: Union[int, Iterable[float]] = 180, circle: bool = True) -> np.ndarray:
    """Compute radon transforms for an in-memory stack (N,H,W).

    Kept for compatibility with previous API.
    """
    if stack.ndim != 3:
        raise ValueError('stack must be a 3D array (N,H,W)')
    theta = _theta_from_angles(angles)
    N = stack.shape[0]
    sample = stack[0]
    sino0 = naive_radon(sample, theta)
    proj_len = sino0.shape[0]
    projections = np.zeros((N, sino0.shape[1], proj_len), dtype=np.float32)
    for i in range(N):
        sino = naive_radon(stack[i], theta)
        projections[i] = sino.T
    return projections
