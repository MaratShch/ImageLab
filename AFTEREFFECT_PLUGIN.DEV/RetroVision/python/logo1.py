from PIL import Image, ImageDraw
import os

# --- Palette Definitions ---

# CGA Palettes
CGA_P0_N_PALETTE = [(0, 0, 0), (0, 170, 170), (170, 0, 170), (255, 255, 255)] # Palette 0 Normal
CGA_P0_I_PALETTE = [(0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255)] # Palette 0 Intense
CGA_P1_N_PALETTE = [(0, 0, 0), (0, 170, 0), (170, 0, 0), (170, 85, 0)] # Palette 1 Normal
CGA_P1_I_PALETTE = [(0, 0, 0), (85, 255, 85), (255, 85, 85), (255, 255, 85)] # Palette 1 Intense

# EGA Palettes
EGA_STANDARD_PALETTE = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
]
PALETTE_KQ3 = [ # King's Quest III Approximation
    (0, 0, 0), (0, 0, 128), (0, 80, 0), (0, 80, 80),
    (128, 0, 0), (128, 0, 128), (128, 64, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 200, 85), (85, 200, 200),
    (200, 85, 85), (200, 85, 200), (255, 255, 85), (255, 255, 255)
]
PALETTE_THEXDER = [ # Thexder Approximation
    (0, 0, 0), (85, 0, 85), (50, 85, 50), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
]
# NEW EGA Palette: Muted/Desaturated variant
EGA_MUTED_PALETTE = [
    (0, 0, 0), (50, 50, 80), (50, 80, 50), (50, 80, 80),
    (80, 50, 50), (80, 50, 80), (80, 64, 50), (100, 100, 100),
    (85, 85, 85), (85, 85, 150), (85, 150, 85), (85, 150, 150),
    (150, 85, 85), (150, 85, 150), (150, 150, 85), (170, 170, 170)
]

LOGO_SIZE = 48

# --- Helper Function to Create PNG ---
def create_png_from_bitmap_data(filename, bitmap_pixels):
    """Creates a PNG file from a 2D array of RGB tuples."""
    height = len(bitmap_pixels)
    width = len(bitmap_pixels[0])

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            pixels[x, y] = bitmap_pixels[y][x]
    
    img.save(filename, "PNG")
    print(f"Saved {filename}")

# --- Function to draw patterned logo ---
def draw_palette_pattern_logo(filename, palette, split_size_x, split_size_y):
    """
    Draws a logo by splitting the area into squares and filling with palette colors.
    """
    bitmap = [[palette[0]] * LOGO_SIZE for _ in range(LOGO_SIZE)] # Start with black background

    num_colors = len(palette)
    color_index = 0

    # Fill the grid with colors from the palette
    for y in range(0, LOGO_SIZE, split_size_y):
        for x in range(0, LOGO_SIZE, split_size_x):
            current_color = palette[color_index % num_colors]
            
            # Draw a square with this color
            for dy in range(split_size_y):
                for dx in range(split_size_x):
                    px, py = x + dx, y + dy
                    if px < LOGO_SIZE and py < LOGO_SIZE:
                        bitmap[py][px] = current_color
            
            color_index += 1

    create_png_from_bitmap_data(filename, bitmap)


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists("retro_logos_pattern_v4"):
        os.makedirs("retro_logos_pattern_v4")

    # --- CGA Logos (48x48 split into 2x2 squares of 24x24) ---
    # CGA P0 Normal
    draw_palette_pattern_logo("retro_logos_pattern_v4/cga_p0_n_pattern.png", CGA_P0_N_PALETTE, 24, 24)
    # CGA P0 Intense
    draw_palette_pattern_logo("retro_logos_pattern_v4/cga_p0_i_pattern.png", CGA_P0_I_PALETTE, 24, 24)
    # CGA P1 Normal
    draw_palette_pattern_logo("retro_logos_pattern_v4/cga_p1_n_pattern.png", CGA_P1_N_PALETTE, 24, 24)
    # CGA P1 Intense
    draw_palette_pattern_logo("retro_logos_pattern_v4/cga_p1_i_pattern.png", CGA_P1_I_PALETTE, 24, 24)

    # --- EGA Logos (48x48 split into 4x4 squares of 12x12) ---
    # EGA Standard
    draw_palette_pattern_logo("retro_logos_pattern_v4/ega_std_pattern.png", EGA_STANDARD_PALETTE, 12, 12)
    # EGA KQ3
    draw_palette_pattern_logo("retro_logos_pattern_v4/ega_kq3_pattern.png", PALETTE_KQ3, 12, 12)
    # EGA Thexder
    draw_palette_pattern_logo("retro_logos_pattern_v4/ega_thexder_pattern.png", PALETTE_THEXDER, 12, 12)
    # NEW: EGA Muted/Desaturated
    draw_palette_pattern_logo("retro_logos_pattern_v4/ega_muted_pattern.png", EGA_MUTED_PALETTE, 12, 12)