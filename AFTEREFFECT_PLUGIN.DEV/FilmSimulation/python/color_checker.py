from PIL import Image, ImageDraw, ImageFont

# Define the classic ColorChecker color values (R, G, B, A)
# These values are calibrated to match the distinct pigments of the standard chart.
patch_colors = [
    # Row 1: Natural Colors
    (115,  82,  68, 255),  (194, 150, 130, 255),  ( 98, 122, 157, 255),  ( 87, 108,  67, 255),  (133, 128, 177, 255),  (103, 189, 170, 255),
    # Row 2: Miscellaneous and Foliage
    (214, 126,  44, 255),  ( 80,  91, 166, 255),  (193,  90,  99, 255),  ( 94,  60, 108, 255),  (157, 188,  64, 255),  (224, 163,  46, 255),
    # Row 3: Primary Colors
    ( 56,  61, 150, 255),  ( 70, 148,  73, 255),  (175,  54,  60, 255),  (231, 199,  31, 255),  (187,  86, 149, 255),  (  8, 133, 161, 255),
    # Row 4: Greyscale
    (243, 243, 242, 255),  (200, 200, 200, 255),  (160, 160, 160, 255),  (122, 122, 121, 255),  ( 85,  85,  85, 255),  ( 52,  52,  52, 255),
]

# Set basic target properties
cols, rows = 6, 4
target_width, target_height = 2400, 1600 # 3:2 aspect ratio
bezel_color = (20, 20, 20, 255) # Matte black border

# Define spacing and dimensions
outer_margin_px = 100
patch_gap_px = 20
total_grid_w = target_width - (2 * outer_margin_px) - ((cols - 1) * patch_gap_px)
total_grid_h = target_height - (2 * outer_margin_px) - ((rows - 1) * patch_gap_px)
patch_w = total_grid_w / cols
patch_h = total_grid_h / rows

# Initialize the 8-bit per channel RGBA canvas
img = Image.new("RGBA", (target_width, target_height), bezel_color)
draw = ImageDraw.Draw(img)

# Main generation loop for patches
for r in range(rows):
    for c in range(cols):
        idx = r * cols + c
        color = patch_colors[idx]
        
        x0 = int(outer_margin_px + c * (patch_w + patch_gap_px))
        y0 = int(outer_margin_px + r * (patch_h + patch_gap_px))
        x1 = int(x0 + patch_w)
        y1 = int(y0 + patch_h)
        
        # Draw the patch as a rounded rectangle for a more accurate simulation
        draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=color, outline=(40, 40, 40, 255), width=2)

# Save the pixel-perfect result as an uncompressed 8-bit RGBA PNG
img.save("classic_colorchecker_pigments.png", format="PNG", compress_level=0)
print("Classic ColorChecker pigments PNG (RGBA 8-bit/channel) generated successfully.")