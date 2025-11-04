"""CLI to process a stack of images and compute Radon projections with streaming/batching."""
import argparse
from pathlib import Path
from src.radon_pipeline import stream_radon


def parse_args():
    p = argparse.ArgumentParser(description="Compute Radon projections for a stack of 2D images (streaming)")
    p.add_argument("--input", required=True, help="Input directory of images or .npy stack")
    p.add_argument("--output", required=True, help="Output .npy file for projections or output directory when --per-image is set")
    p.add_argument("--angles", nargs='+', type=float, help="Angles in degrees (space separated). If omitted, 180 angles are used.")
    p.add_argument("--batch-size", type=int, default=32, help="Number of images to process per batch")
    p.add_argument("--per-image", action='store_true', help="Save one .npy per input image into output directory")
    p.add_argument("--circle", action='store_true', help="Pass circle=True to skimage.radon (default False)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.angles:
        if len(args.angles) == 1:
            angles = int(args.angles[0])
        else:
            angles = args.angles
    else:
        angles = 180

    out = stream_radon(args.input, args.output, angles=angles, batch_size=args.batch_size, per_image=args.per_image, circle=args.circle)
    if out is not None:
        print(f"Wrote projections to {out}")
    else:
        print(f"Saved per-image projections to directory {args.output}")


if __name__ == '__main__':
    main()
