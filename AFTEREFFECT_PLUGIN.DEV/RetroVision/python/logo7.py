from PIL import Image

# --- Define standard VGA 16-color palette (RGB) ---
vga16_palette = [
    (0, 0, 0),          # 0: Black
    (0, 0, 170),        # 1: Blue
    (0, 170, 0),        # 2: Green
    (0, 170, 170),      # 3: Cyan
    (170, 0, 0),        # 4: Red
    (170, 0, 170),      # 5: Magenta
    (170, 85, 0),       # 6: Brown
    (170, 170, 170),    # 7: Light Gray
    (85, 85, 85),       # 8: Dark Gray
    (85, 85, 255),      # 9: Bright Blue
    (85, 255, 85),      # 10: Bright Green
    (85, 255, 255),     # 11: Bright Cyan
    (255, 85, 85),      # 12: Bright Red
    (255, 85, 255),     # 13: Bright Magenta
    (255, 255, 85),     # 14: Yellow
    (255, 255, 255),    # 15: White
]

# --- Create 48x48 RGBA image ---
width, height = 48, 48
img = Image.new("RGBA", (width, height))

# --- Draw repeating color blocks (3x3 pixel per color) ---
block_size = 12  # 4 blocks across → 4x4 grid = 16 colors total
for i, color in enumerate(vga16_palette):
    x0 = (i % 4) * block_size
    y0 = (i // 4) * block_size
    for y in range(y0, y0 + block_size):
        for x in range(x0, x0 + block_size):
            img.putpixel((x, y), color + (255,))  # Add alpha=255

# --- Fill entire image by tiling the 4x4 pattern ---
pixels = img.load()
for y in range(height):
    for x in range(width):
        src_x = x % (block_size * 4)
        src_y = y % (block_size * 4)
        img.putpixel((x, y), pixels[src_x, src_y])

# --- Save result as 32-bit PNG ---
img.save("vga16_palette_48x48.png", "PNG")
print("Saved: vga16_palette_48x48.png")
