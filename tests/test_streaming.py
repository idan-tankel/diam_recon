import numpy as np
import os
from src.radon_pipeline import stream_radon


def make_circle(h, w, radius=None):
    if radius is None:
        radius = min(h, w) // 4
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask = (Y - cy)**2 + (X - cx)**2 <= radius**2
    return mask.astype(np.float32)


def test_stream_radon_npy(tmp_path):
    N = 8
    H = W = 64
    stack = np.stack([make_circle(H, W) for _ in range(N)], axis=0)
    in_path = tmp_path / 'stack.npy'
    out_path = tmp_path / 'projs.npy'
    np.save(str(in_path), stack)

    result = stream_radon(str(in_path), str(out_path), angles=36, batch_size=3)
    assert result is not None
    projs = np.load(str(out_path), mmap_mode='r')
    assert projs.shape[0] == N
    assert projs.shape[1] == 36


def test_stream_radon_dir_per_image(tmp_path):
    N = 6
    H = W = 64
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    for i in range(N):
        arr = make_circle(H, W)
        np.save(str(img_dir / f'img_{i}.npy'), arr)  # write as .npy and load via imageio will fail, so write as png
    # To ensure we have image files, write pngs instead
    import imageio
    for i in range(N):
        arr = make_circle(H, W)
        imageio.v2.imwrite(str(img_dir / f'{i:03d}.png'), (arr * 255).astype('uint8'))

    out_dir = tmp_path / 'out'
    stream_radon(str(img_dir), str(out_dir), angles=18, batch_size=2, per_image=True)
    files = sorted(os.listdir(str(out_dir)))
    assert len(files) == N
