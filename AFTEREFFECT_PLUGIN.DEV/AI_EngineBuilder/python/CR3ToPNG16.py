import rawpy
import numpy as np
import png
import sys
import os

def cr3_to_png_16bit(input_path, output_path):
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            gamma=(1, 1)
        )

    rgb = np.asarray(rgb, dtype=np.uint16)

    height, width, channels = rgb.shape
    assert channels == 3

    # PyPNG expects rows as flat sequences: [R,G,B,R,G,B,...]
    rgb_2d = rgb.reshape(height, width * 3)

    with open(output_path, 'wb') as f:
        writer = png.Writer(
            width=width,
            height=height,
            bitdepth=16,
            greyscale=False
        )
        writer.write(f, rgb_2d)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    cr3_to_png_16bit(sys.argv[1], sys.argv[2])