import rawpy
import numpy as np
import png
import sys
import os

# --- The core function you provided ---
# (Maintained without changes as requested)
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


# --- MODIFIED: Main section for BATCH processing ---
if __name__ == "__main__":
    # Get the current directory (level 1 only)
    current_dir = os.getcwd()
    print(f"Scanning directory for CR3 files: {current_dir}")

    # Identify all files ending in .CR3 (case-insensitive for robustness)
    cr3_files = []
    for file in os.listdir(current_dir):
        if file.lower().endswith('.cr3'):
            # Only add the filename, not the full path for now.
            cr3_files.append(file)

    # If no files are found, exit gracefully
    if not cr3_files:
        print("No .CR3 files found in the current directory.")
        sys.exit(0)

    print(f"Found {len(cr3_files)} .CR3 files. Starting conversion...")

    # Iterate through the found CR3 files
    count = 0
    for cr3_filename in cr3_files:
        try:
            # Construct input and output paths
            # Input path is just the filename in the current dir.
            input_path = cr3_filename

            # Output path is the same base name, but with .png
            base_name, _ = os.path.splitext(cr3_filename)
            output_path = base_name + '.png'

            # Execute the conversion
            cr3_to_png_16bit(input_path, output_path)
            count += 1
        except Exception as e:
            # Handle potential errors with a specific file so the batch isn't halted
            print(f"Error converting {cr3_filename}: {e}")

    print(f"Batch conversion complete. Total converted: {count}")