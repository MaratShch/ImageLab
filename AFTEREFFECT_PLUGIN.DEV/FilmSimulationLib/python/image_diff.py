#!/usr/bin/env python3
"""Absolute per-pixel, per-channel difference between two PNG images.

Usage
-----
    python image_diff.py <image_one.png> <image_two.png>

Writes ``image_diff.png`` next to the current working directory and prints a
per-channel summary of the largest difference found.

Why this module decodes PNG itself
----------------------------------
It would be shorter to hand the files to Pillow. Measured on Pillow 12.2, that
silently destroys the data this script exists to measure:

    16-bit RGB PNG, pixel value (65535, 1000, 7)
        Pillow returns  uint8  (255, 3, 0)
        imageio         uint8  (255, 3, 0)      -- same Pillow backend
        OpenCV          uint16 (65535, 1000, 7) -- correct, but BGR and a heavy
                                                   dependency

Pillow reads 16-bit GREYSCALE correctly, as mode ``I;16``, which makes the
failure worse than a plain limitation: whether precision survives depends on the
colour type of the file, so the bug appears and disappears with the input. A tool
whose whole purpose is to detect a one-least-significant-bit difference cannot be
built on a loader that discards the low byte of every sample without saying so.

So the PNG decoding here is done with the standard library only: ``zlib`` for the
compressed stream and explicit un-filtering of each scanline. That is roughly
sixty lines, it has no dependency beyond numpy for the arithmetic, and it is
exact at both 8 and 16 bits.

Deliberate limitations, each of which raises a clear error rather than guessing:

    * Interlaced (Adam7) files are rejected. Un-interlacing correctly is a lot of
      code for a case no rendering pipeline emits.
    * Palette images (colour type 3) are rejected. A palette index is not a
      colour value, so differencing indices is meaningless, and expanding the
      palette would silently change what "channel" means.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PNG format constants
# ---------------------------------------------------------------------------

#: The eight bytes every PNG file begins with (PNG specification, clause 5.2).
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Number of colour/alpha samples carried per pixel, indexed by PNG colour type.
#: Colour type 3 is palette and is deliberately absent -- see the module docstring.
SAMPLES_PER_PIXEL_BY_COLOUR_TYPE = {
    0: 1,  # greyscale
    2: 3,  # truecolour, R G B
    4: 2,  # greyscale with alpha
    6: 4,  # truecolour with alpha, R G B A
}

#: Human-readable channel names, indexed by PNG colour type. Used for the report,
#: so a greyscale file is never labelled with red, green and blue channels it does
#: not have.
CHANNEL_NAMES_BY_COLOUR_TYPE = {
    0: ("Grey",),
    2: ("R", "G", "B"),
    4: ("Grey", "Alpha"),
    6: ("R", "G", "B", "Alpha"),
}

#: Colour types whose last sample is an alpha channel rather than a colour.
COLOUR_TYPES_WITH_ALPHA = frozenset({4, 6})

#: Readable name of each pixel format, for messages. Kept separate from
#: CHANNEL_NAMES_BY_COLOUR_TYPE because concatenating those names produces
#: unreadable labels such as "RGBAlpha".
FORMAT_NAME_BY_COLOUR_TYPE = {
    0: "greyscale",
    2: "RGB",
    4: "greyscale + alpha",
    6: "RGB + alpha",
}

#: Name of the file written by this script.
OUTPUT_FILENAME = "image_diff.png"

#: Process exit status used for every handled error.
EXIT_FAILURE = 1


class ImageDiffError(Exception):
    """Any condition that should stop the script with a message, not a traceback.

    Every raise site is a situation the user can act on: a missing file, a
    malformed PNG, a size mismatch. Reserving a single exception type for these
    keeps ``main`` able to distinguish an expected failure from a genuine bug,
    which should still produce a traceback.
    """


# ---------------------------------------------------------------------------
# Decoded image
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PngImage:
    """One decoded PNG image and the header fields that describe its format.

    Attributes:
        path: Where the image came from, used in error messages.
        width: Image width in pixels.
        height: Image height in pixels.
        bit_depth: Bits per sample, 8 or 16.
        colour_type: PNG colour type from the IHDR chunk.
        samples: Decoded samples, shaped ``(height, width, samples_per_pixel)``.
            The dtype is ``uint8`` or ``uint16`` matching ``bit_depth`` exactly --
            no promotion, no scaling, no rounding.
    """

    path: Path
    width: int
    height: int
    bit_depth: int
    colour_type: int
    samples: np.ndarray

    @property
    def samples_per_pixel(self) -> int:
        """Number of samples carried per pixel, including alpha if present."""
        return SAMPLES_PER_PIXEL_BY_COLOUR_TYPE[self.colour_type]

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Names of the samples, in storage order."""
        return CHANNEL_NAMES_BY_COLOUR_TYPE[self.colour_type]

    @property
    def colour_channel_count(self) -> int:
        """Number of channels that carry colour, excluding any alpha channel."""
        return self.samples_per_pixel - (1 if self.colour_type in COLOUR_TYPES_WITH_ALPHA else 0)

    def format_description(self) -> str:
        """A short description of the pixel format, for messages."""
        name = FORMAT_NAME_BY_COLOUR_TYPE[self.colour_type]
        return f"{name}, {self.bit_depth}-bit (PNG colour type {self.colour_type})"


# ---------------------------------------------------------------------------
# PNG decoding
# ---------------------------------------------------------------------------

def _iterate_chunks(data: bytes, path: Path):
    """Yield ``(chunk_type, chunk_payload)`` for each chunk in a PNG file.

    The CRC of every chunk is verified. A corrupted file is far more useful to
    report as corrupted than to decode into plausible-looking noise, and a diff
    tool that accepts damaged input will eventually be used to "prove" a
    difference that is really disc rot.
    """
    offset = len(PNG_SIGNATURE)
    total_length = len(data)

    while offset < total_length:
        # Each chunk is: 4-byte payload length, 4-byte type, payload, 4-byte CRC.
        if offset + 8 > total_length:
            raise ImageDiffError(f"{path}: truncated PNG, incomplete chunk header")

        (payload_length,) = struct.unpack(">I", data[offset:offset + 4])
        chunk_type = data[offset + 4:offset + 8]

        payload_start = offset + 8
        payload_end = payload_start + payload_length
        crc_end = payload_end + 4

        if crc_end > total_length:
            raise ImageDiffError(
                f"{path}: truncated PNG, chunk {chunk_type.decode('ascii', 'replace')} "
                f"claims {payload_length} bytes but the file ends first"
            )

        payload = data[payload_start:payload_end]

        # The CRC covers the chunk type and the payload, but not the length field.
        (stored_crc,) = struct.unpack(">I", data[payload_end:crc_end])
        computed_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF

        if stored_crc != computed_crc:
            raise ImageDiffError(
                f"{path}: corrupted PNG, CRC mismatch in chunk "
                f"{chunk_type.decode('ascii', 'replace')}"
            )

        yield chunk_type, payload

        offset = crc_end


def _undo_scanline_filters(
    filtered_rows: bytes,
    height: int,
    row_length_bytes: int,
    bytes_per_pixel: int,
    path: Path,
) -> bytearray:
    """Reverse the per-scanline filtering that PNG applies before compression.

    PNG prefixes every scanline with a filter byte and stores differences rather
    than values, because differences compress far better. The five filters are
    defined in clause 9.2 of the specification and each is undone by adding back
    the predictor it subtracted.

    Args:
        filtered_rows: The decompressed IDAT stream: ``height`` repetitions of one
            filter byte followed by ``row_length_bytes`` of filtered data.
        height: Number of scanlines.
        row_length_bytes: Bytes per scanline, excluding the filter byte.
        bytes_per_pixel: Distance in bytes to the sample to the left. Filters work
            on whole pixels, not samples, so this is the stride the ``a`` and
            ``c`` predictors use.
        path: Source path, for error messages.

    Returns:
        The unfiltered raw samples, ``height * row_length_bytes`` bytes.
    """
    expected_length = height * (row_length_bytes + 1)

    if len(filtered_rows) < expected_length:
        raise ImageDiffError(
            f"{path}: PNG image data is short -- expected {expected_length} bytes "
            f"after decompression, found {len(filtered_rows)}"
        )

    raw = bytearray(height * row_length_bytes)

    # A row of zeros standing in for the scanline above the first one, so the
    # first row needs no special case.
    previous_row = bytearray(row_length_bytes)

    source_offset = 0

    for row_index in range(height):
        filter_type = filtered_rows[source_offset]
        source_offset += 1

        current_row = bytearray(
            filtered_rows[source_offset:source_offset + row_length_bytes]
        )
        source_offset += row_length_bytes

        if filter_type == 0:
            # None: the bytes are already the values.
            pass

        elif filter_type == 1:
            # Sub: each byte is a difference from the byte one pixel to its left.
            for i in range(bytes_per_pixel, row_length_bytes):
                current_row[i] = (current_row[i] + current_row[i - bytes_per_pixel]) & 0xFF

        elif filter_type == 2:
            # Up: each byte is a difference from the byte directly above it.
            for i in range(row_length_bytes):
                current_row[i] = (current_row[i] + previous_row[i]) & 0xFF

        elif filter_type == 3:
            # Average: predictor is the mean of left and above, rounded DOWN. The
            # floor is part of the specification; rounding to nearest here shifts
            # every subsequent byte in the row.
            for i in range(row_length_bytes):
                left = current_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                above = previous_row[i]
                current_row[i] = (current_row[i] + ((left + above) >> 1)) & 0xFF

        elif filter_type == 4:
            # Paeth: predictor is whichever of left, above and upper-left is
            # closest to left + above - upper_left.
            for i in range(row_length_bytes):
                left = current_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                above = previous_row[i]
                upper_left = previous_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0

                estimate = left + above - upper_left

                distance_left = abs(estimate - left)
                distance_above = abs(estimate - above)
                distance_upper_left = abs(estimate - upper_left)

                # Ties resolve towards left, then above -- also specified, and also
                # a source of silent corruption if reordered.
                if distance_left <= distance_above and distance_left <= distance_upper_left:
                    predictor = left
                elif distance_above <= distance_upper_left:
                    predictor = above
                else:
                    predictor = upper_left

                current_row[i] = (current_row[i] + predictor) & 0xFF

        else:
            raise ImageDiffError(
                f"{path}: PNG uses unknown scanline filter {filter_type} "
                f"on row {row_index}"
            )

        destination = row_index * row_length_bytes
        raw[destination:destination + row_length_bytes] = current_row

        previous_row = current_row

    return raw


def read_png(path: Path) -> PngImage:
    """Decode a PNG file into a ``PngImage`` at its full native bit depth.

    Raises:
        ImageDiffError: If the file is missing, unreadable, not a PNG, malformed,
            or uses a feature this decoder deliberately does not support.
    """
    try:
        file_bytes = path.read_bytes()
    except FileNotFoundError:
        raise ImageDiffError(f"{path}: file does not exist") from None
    except IsADirectoryError:
        raise ImageDiffError(f"{path}: is a directory, not a file") from None
    except PermissionError:
        raise ImageDiffError(f"{path}: permission denied") from None
    except OSError as error:
        raise ImageDiffError(f"{path}: cannot be read ({error})") from None

    if not file_bytes.startswith(PNG_SIGNATURE):
        raise ImageDiffError(
            f"{path}: not a PNG file -- the eight-byte PNG signature is missing"
        )

    header_seen = False
    width = height = bit_depth = colour_type = 0
    compressed_parts: list[bytes] = []

    for chunk_type, payload in _iterate_chunks(file_bytes, path):
        if chunk_type == b"IHDR":
            if len(payload) != 13:
                raise ImageDiffError(
                    f"{path}: malformed PNG, IHDR must be 13 bytes, found {len(payload)}"
                )

            (width, height, bit_depth, colour_type,
             compression_method, filter_method, interlace_method) = struct.unpack(
                ">IIBBBBB", payload
            )

            if width == 0 or height == 0:
                raise ImageDiffError(f"{path}: malformed PNG, zero width or height")

            if compression_method != 0:
                raise ImageDiffError(
                    f"{path}: PNG compression method {compression_method} is not defined "
                    f"by the specification"
                )

            if filter_method != 0:
                raise ImageDiffError(
                    f"{path}: PNG filter method {filter_method} is not defined by the "
                    f"specification"
                )

            if interlace_method != 0:
                raise ImageDiffError(
                    f"{path}: interlaced PNG is not supported. Re-save the file "
                    f"without Adam7 interlacing."
                )

            if colour_type == 3:
                raise ImageDiffError(
                    f"{path}: palette PNG is not supported. A palette index is not a "
                    f"colour value, so differencing indices would be meaningless. "
                    f"Re-save as greyscale or truecolour."
                )

            if colour_type not in SAMPLES_PER_PIXEL_BY_COLOUR_TYPE:
                raise ImageDiffError(
                    f"{path}: PNG colour type {colour_type} is not defined by the "
                    f"specification"
                )

            if bit_depth not in (8, 16):
                raise ImageDiffError(
                    f"{path}: {bit_depth}-bit PNG is not supported, only 8 and 16. "
                    f"Sub-byte depths pack several pixels per byte and are not a "
                    f"format any render pipeline produces."
                )

            header_seen = True

        elif chunk_type == b"IDAT":
            # Image data may be split across any number of IDAT chunks, which must
            # be concatenated before decompression -- the deflate stream spans them.
            compressed_parts.append(payload)

        elif chunk_type == b"IEND":
            break

        # Every other chunk is ancillary to a pixel difference and is skipped.

    if not header_seen:
        raise ImageDiffError(f"{path}: malformed PNG, no IHDR chunk found")

    if not compressed_parts:
        raise ImageDiffError(f"{path}: malformed PNG, no IDAT image data found")

    try:
        filtered_rows = zlib.decompress(b"".join(compressed_parts))
    except zlib.error as error:
        raise ImageDiffError(
            f"{path}: corrupted PNG, image data will not decompress ({error})"
        ) from None

    samples_per_pixel = SAMPLES_PER_PIXEL_BY_COLOUR_TYPE[colour_type]
    bytes_per_sample = bit_depth // 8
    bytes_per_pixel = samples_per_pixel * bytes_per_sample
    row_length_bytes = width * bytes_per_pixel

    raw = _undo_scanline_filters(
        filtered_rows, height, row_length_bytes, bytes_per_pixel, path
    )

    # PNG stores multi-byte samples most significant byte first, regardless of the
    # host's own byte order, hence the explicit big-endian dtype.
    dtype = np.dtype(">u2") if bit_depth == 16 else np.dtype("u1")

    samples = np.frombuffer(bytes(raw), dtype=dtype)
    samples = samples.reshape(height, width, samples_per_pixel)

    # Convert to the host's native order so the arithmetic below is not doing a
    # byte swap on every access.
    samples = np.ascontiguousarray(samples, dtype=np.uint16 if bit_depth == 16 else np.uint8)

    return PngImage(
        path=path,
        width=width,
        height=height,
        bit_depth=bit_depth,
        colour_type=colour_type,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------

def _make_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Assemble one PNG chunk: length, type, payload, CRC."""
    type_and_payload = chunk_type + payload

    return (
        struct.pack(">I", len(payload))
        + type_and_payload
        + struct.pack(">I", zlib.crc32(type_and_payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, samples: np.ndarray, bit_depth: int, colour_type: int) -> None:
    """Write a non-interlaced, unfiltered PNG at the given bit depth.

    Filter type 0 -- no filtering -- on every scanline. A difference image is
    mostly zeros with isolated spikes, which deflate already compresses well, so
    predictive filtering would add code for very little file size.

    Args:
        path: Destination file.
        samples: ``(height, width, samples_per_pixel)`` array whose dtype matches
            ``bit_depth``.
        bit_depth: 8 or 16.
        colour_type: PNG colour type describing ``samples``.
    """
    height, width, _ = samples.shape

    # Back to big-endian for storage if the samples are multi-byte.
    stored = samples.astype(">u2" if bit_depth == 16 else "u1", copy=False)

    # Every scanline is prefixed with its filter byte, here always zero.
    scanlines = bytearray()

    for row_index in range(height):
        scanlines.append(0)
        scanlines.extend(stored[row_index].tobytes())

    header = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        colour_type,
        0,  # compression method: deflate, the only one defined
        0,  # filter method: adaptive, the only one defined
        0,  # interlace method: none
    )

    png_bytes = (
        PNG_SIGNATURE
        + _make_chunk(b"IHDR", header)
        + _make_chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
        + _make_chunk(b"IEND", b"")
    )

    try:
        path.write_bytes(png_bytes)
    except OSError as error:
        raise ImageDiffError(f"{path}: cannot be written ({error})") from None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def verify_comparable(image_one: PngImage, image_two: PngImage) -> None:
    """Check that two images can be differenced at all.

    Raises:
        ImageDiffError: If the dimensions or the pixel formats differ.
    """
    if (image_one.width, image_one.height) != (image_two.width, image_two.height):
        raise ImageDiffError(
            "images have different dimensions:\n"
            f"    {image_one.path}: {image_one.width} x {image_one.height}\n"
            f"    {image_two.path}: {image_two.width} x {image_two.height}"
        )

    if (image_one.bit_depth, image_one.colour_type) != (
        image_two.bit_depth,
        image_two.colour_type,
    ):
        raise ImageDiffError(
            "images have incompatible colour formats:\n"
            f"    {image_one.path}: {image_one.format_description()}\n"
            f"    {image_two.path}: {image_two.format_description()}"
        )


@dataclass(frozen=True)
class ChannelSummary:
    """The largest absolute difference found in one channel.

    Attributes:
        name: Channel name, such as ``"R"``.
        maximum_difference: Largest absolute difference in this channel.
        position_x: Column of that difference, zero based.
        position_y: Row of that difference, zero based.
        value_one: Sample value at that position in the first image.
        value_two: Sample value at that position in the second image.
    """

    name: str
    maximum_difference: int
    position_x: int
    position_y: int
    value_one: int
    value_two: int


def compute_absolute_difference(
    image_one: PngImage, image_two: PngImage
) -> tuple[np.ndarray, list[ChannelSummary]]:
    """Compute the absolute difference and summarise the worst pixel per channel.

    The subtraction is done in a SIGNED wide type. Subtracting one unsigned array
    from another wraps around instead of going negative -- for ``uint8``,
    ``10 - 20`` is 246, not -10 -- so a naive difference reports a huge error
    wherever the second image is merely brighter. Widening first, taking the
    absolute value, then narrowing back is what makes the result correct.

    Args:
        image_one: First image; the one reported as "Image #1".
        image_two: Second image, already verified comparable.

    Returns:
        A pair of ``(difference_samples, summaries)``. The difference array has the
        same shape and dtype as the inputs. Any alpha channel in the output is set
        to fully opaque rather than differenced -- see the note in the body.
    """
    # int32 holds the full signed range of both uint8 and uint16 differences with
    # room to spare, so there is no possibility of overflow here.
    difference_wide = np.abs(
        image_one.samples.astype(np.int32) - image_two.samples.astype(np.int32)
    )

    difference = difference_wide.astype(image_one.samples.dtype)

    colour_channel_count = image_one.colour_channel_count

    # ------------------------------------------------------------------
    #  Alpha is forced opaque rather than differenced.
    #
    #  Two renders normally share an alpha channel, so its difference is zero
    #  everywhere -- and an RGBA PNG with zero alpha is fully TRANSPARENT, which
    #  means the difference image would open as a blank rectangle and every real
    #  difference in the colour channels would be invisible. Differencing alpha is
    #  the mathematically consistent choice and the practically useless one.
    # ------------------------------------------------------------------
    if image_one.colour_type in COLOUR_TYPES_WITH_ALPHA:
        opaque_value = 65535 if image_one.bit_depth == 16 else 255
        difference[:, :, colour_channel_count:] = opaque_value

    summaries: list[ChannelSummary] = []

    for channel_index in range(colour_channel_count):
        channel_difference = difference_wide[:, :, channel_index]

        # argmax on the flattened array returns the FIRST maximum in raster order,
        # so a tie is reported at the topmost, then leftmost, position. Documented
        # because otherwise the reported coordinate looks arbitrary when a
        # difference is uniform over a region.
        flat_index = int(np.argmax(channel_difference))
        position_y, position_x = np.unravel_index(flat_index, channel_difference.shape)

        summaries.append(
            ChannelSummary(
                name=image_one.channel_names[channel_index],
                maximum_difference=int(channel_difference[position_y, position_x]),
                position_x=int(position_x),
                position_y=int(position_y),
                value_one=int(image_one.samples[position_y, position_x, channel_index]),
                value_two=int(image_two.samples[position_y, position_x, channel_index]),
            )
        )

    return difference, summaries


def print_summaries(summaries: list[ChannelSummary]) -> None:
    """Print the per-channel report to standard output."""
    for summary in summaries:
        print(f"Channel {summary.name}")
        print(f"    Maximum Difference : {summary.maximum_difference}")
        print(f"    Position           : ({summary.position_x}, {summary.position_y})")
        print(f"    Image #1 Value     : {summary.value_one}")
        print(f"    Image #2 Value     : {summary.value_two}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    argparse exits with status 2 by itself when a required argument is missing,
    after printing a usage message, so the missing-argument case needs no code of
    its own here.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the absolute per-pixel, per-channel difference between two "
            "PNG images."
        ),
        epilog=f"Writes {OUTPUT_FILENAME} in the current directory.",
    )

    parser.add_argument("image_one", type=Path, help="full path to input image #1")
    parser.add_argument("image_two", type=Path, help="full path to input image #2")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(OUTPUT_FILENAME),
        help=f"difference image to write (default: {OUTPUT_FILENAME})",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison. Returns the process exit status."""
    arguments = parse_arguments(argv)

    try:
        image_one = read_png(arguments.image_one)
        image_two = read_png(arguments.image_two)

        verify_comparable(image_one, image_two)

        difference, summaries = compute_absolute_difference(image_one, image_two)

        write_png(
            arguments.output,
            difference,
            image_one.bit_depth,
            image_one.colour_type,
        )

    except ImageDiffError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(
        f"{image_one.width} x {image_one.height}, {image_one.format_description()}"
    )
    print(f"Difference image written to {arguments.output}")
    print()

    print_summaries(summaries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
