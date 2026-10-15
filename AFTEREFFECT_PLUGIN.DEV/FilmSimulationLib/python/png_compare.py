#!/usr/bin/env python3
"""png_compare.py -- absolute per-channel comparison of two PNG images.

Compares two PNG files pixel by pixel, reports the per-channel maxima and
where they occur, and writes a PNG of the absolute differences.

USAGE
    python png_compare.py <first_file_name> <second_file_name> [--stretch]

    <first_file_name>   mandatory, path to the first PNG
    <second_file_name>  mandatory, path to the second PNG
    --stretch           optional, scale the difference image to full range

COORDINATE CONVENTION
    Coordinates are reported as (x, y) with (0, 0) at the TOP-LEFT corner:
    x runs left to right across the image, y runs top to bottom.  This is
    the opposite order from NumPy's own array indexing, which is [row,
    column] = [y, x], and the conversion is done explicitly at every point
    where a coordinate is printed.

THE COMPARISON
    For every pixel and every channel:

        difference = abs(int(image1) - int(image2))

    The subtraction is done in a signed integer type before the absolute
    value is taken.  Doing it in the images' own uint8 would wrap around --
    5 - 250 would come out as 11 instead of 245 -- so the arrays are widened
    to int16 first.  This is the single most important detail in the file.

THE DIFFERENCE IMAGE
    Without --stretch the output PNG contains the absolute differences
    EXACTLY as calculated and nothing else.  A pixel differing by R=1, G=16,
    B=7 is written as R=1, G=16, B=7.  No normalisation, no scaling, no
    gamma, no tone curve.  The image will usually look almost black, because
    small differences ARE almost black; that is the point of the mode.

    With --stretch every difference is multiplied by

        scale = 255 / maximum_absolute_difference

    where the maximum is taken across all pixels AND all compared channels,
    so the relative weight of the three channels is preserved -- one scale
    factor for the whole image, not one per channel.  With a maximum of 16,
    the R=1, G=16, B=7 pixel above becomes R=16, G=255, B=112.  Results are
    rounded half-to-even and clamped to 0..255.

    If the two images are identical the maximum is zero.  That case is
    handled explicitly: the scale factor is not computed at all and an
    all-black difference image is written, so there is no division by zero.

WHAT THIS SCRIPT WILL NOT DO
    It never resizes, resamples, crops, colour-corrects or otherwise alters
    either input.  Images of differing dimensions are reported as an error
    rather than being made to fit, because silently resampling one of them
    would compare interpolated pixels that exist in neither original.

REQUIREMENTS
    Python 3.12, Pillow, NumPy.  Nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment problem, not logic
    sys.stderr.write(
        "error: Pillow is required. Install it with: pip install Pillow\n")
    raise SystemExit(2)


#: Channel names in array order, used for both reporting and iteration.
CHANNELS: tuple[str, str, str] = ("Red", "Green", "Blue")

#: Maximum value of an 8-bit channel. The stretch target and the output clamp.
CHANNEL_MAX: int = 255

#: Exit codes. 0 = identical, 1 = images differ, 2 = usage or input error.
EXIT_IDENTICAL: int = 0
EXIT_DIFFERENT: int = 1
EXIT_ERROR: int = 2


def fail(message: str) -> None:
    """Print an error to stderr and exit with the input-error code."""
    sys.stderr.write(f"error: {message}\n")
    raise SystemExit(EXIT_ERROR)


def load_png(path_text: str, label: str) -> np.ndarray:
    """Load one PNG and return it as an (h, w, 3) uint8 array.

    Validates that the path exists, is a file, and is actually a PNG --
    the extension is not trusted, Pillow's own format detection is.

    An image with an alpha channel keeps only its RGB planes: the request
    is for a per-channel R/G/B comparison, and silently compositing the
    alpha against an invented background would change the pixel values
    being compared. Greyscale and palette images are expanded to RGB so
    that a greyscale and an RGB rendering of the same picture can still be
    compared; that expansion copies values, it does not alter them.
    """
    path = Path(path_text)

    if not path.exists():
        fail(f"{label} file does not exist: {path}")
    if not path.is_file():
        fail(f"{label} path is not a file: {path}")

    try:
        image = Image.open(path)
        image.load()
    except OSError as exc:
        fail(f"{label} file could not be read as an image: {path} ({exc})")

    if image.format != "PNG":
        fail(f"{label} file is not a PNG (detected {image.format}): {path}")

    if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "I;16", "1"):
        fail(f"{label} file has an unsupported image mode {image.mode!r}: "
             f"{path}")

    if image.mode != "RGB":
        # "P" carries a palette and may hide transparency; converting via
        # RGBA first and then dropping alpha resolves the palette correctly.
        if image.mode in ("P", "RGBA", "LA"):
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")

    array = np.asarray(image, dtype=np.uint8)

    if array.ndim != 3 or array.shape[2] != 3:
        fail(f"{label} file did not yield three channels: {path}")

    return array


def locate_max(plane: np.ndarray) -> tuple[int, int, int]:
    """Return (value, x, y) for the first largest element of a 2-D plane.

    `argmax` returns a flat index into a row-major array, so `divmod` by the
    width recovers (row, column) = (y, x). Ties go to the first occurrence
    in raster order, which makes the report reproducible.
    """
    height, width = plane.shape
    flat_index = int(plane.argmax())
    y, x = divmod(flat_index, width)
    return int(plane[y, x]), int(x), int(y)


def build_difference_image(diff: np.ndarray, stretch: bool,
                           overall_max: int) -> np.ndarray:
    """Turn the int16 difference array into a uint8 image.

    Without `stretch` the values pass through untouched -- they are already
    in 0..255 because the absolute difference of two uint8 values cannot
    exceed 255.

    With `stretch` every channel is multiplied by one common scale factor,
    255 / overall_max, so that the largest difference present becomes
    exactly 255 and the relative weight between channels is preserved.
    """
    if not stretch:
        # abs(a - b) for two uint8 inputs is in 0..255 by construction, so
        # this narrowing cannot overflow. Asserted rather than assumed.
        assert diff.min() >= 0 and diff.max() <= CHANNEL_MAX
        return diff.astype(np.uint8)

    if overall_max <= 0:
        # Identical images. No scale factor is computed at all, which is how
        # the division by zero is avoided rather than patched afterwards.
        return np.zeros_like(diff, dtype=np.uint8)

    scale = CHANNEL_MAX / float(overall_max)
    scaled = np.rint(diff.astype(np.float64) * scale)
    return np.clip(scaled, 0, CHANNEL_MAX).astype(np.uint8)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Build and run the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="png_compare.py",
        description="Compare two PNG images channel by channel and write a "
                    "PNG of the absolute differences.",
        epilog="Coordinates are reported as (x, y) with (0, 0) at the "
               "top-left corner.")
    parser.add_argument("first_file_name",
                        help="path to the first PNG image")
    parser.add_argument("second_file_name",
                        help="path to the second PNG image")
    parser.add_argument("--stretch", action="store_true",
                        help="scale the difference image so that the largest "
                             "difference present becomes 255")
    parser.add_argument("-o", "--output", default=None,
                        help="path for the difference PNG "
                             "(default: <first>_diff.png beside the first "
                             "image)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    first = load_png(args.first_file_name, "first")
    second = load_png(args.second_file_name, "second")

    if first.shape != second.shape:
        fail(
            "image dimensions differ and this script will not resize either "
            f"one: first is {first.shape[1]}x{first.shape[0]}, second is "
            f"{second.shape[1]}x{second.shape[0]} (width x height). Compare "
            "images of the same size, or resize one yourself and be explicit "
            "about it.")

    height, width, _ = first.shape

    # ⚠ WIDEN BEFORE SUBTRACTING. In uint8 the subtraction wraps: 5 - 250
    # would give 11 rather than 245, and every large difference in the image
    # would be reported as a small one. int16 holds the full -255..255 range.
    diff = np.abs(first.astype(np.int16) - second.astype(np.int16))

    total_pixels = height * width
    # A pixel counts as differing if ANY of its three channels differs.
    differing_pixels = int(np.count_nonzero(diff.any(axis=2)))
    overall_max = int(diff.max())

    print(f"Image size:                 {width} x {height} (width x height)")
    print(f"Total pixels:               {total_pixels}")
    print(f"Pixels that differ:         {differing_pixels}")

    for index, name in enumerate(CHANNELS):
        value, x, y = locate_max(diff[:, :, index])
        print(f"Max {name:<5} difference:    {value:<5} at (x={x}, y={y})")

    if args.output is not None:
        out_path = Path(args.output)
    else:
        first_path = Path(args.first_file_name)
        out_path = first_path.with_name(first_path.stem + "_diff.png")

    image = build_difference_image(diff, args.stretch, overall_max)

    if args.stretch:
        if overall_max > 0:
            print(f"Stretch:                    on, scale = 255 / "
                  f"{overall_max} = {CHANNEL_MAX / overall_max:.6f}")
        else:
            print("Stretch:                    on, but the images are "
                  "identical, so no scaling was applied")
    else:
        print("Stretch:                    off, differences written exactly")

    try:
        Image.fromarray(image, mode="RGB").save(out_path, format="PNG")
    except OSError as exc:
        fail(f"the difference image could not be written to {out_path}: {exc}")

    print(f"Difference image:           {out_path}")

    return EXIT_IDENTICAL if differing_pixels == 0 else EXIT_DIFFERENT


if __name__ == "__main__":
    raise SystemExit(main())
