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
EGA_DUNE_PALETTE = [ # Dune-Inspired
    (0, 0, 0), (0, 0, 80), (40, 80, 40), (0, 80, 80),
    (120, 40, 40), (120, 40, 120), (140, 100, 60), (100, 100, 100),
    (80, 80, 80), (80, 80, 160), (80, 120, 80), (80, 120, 120),
    (200, 80, 60), (160, 80, 160), (200, 160, 80), (220, 200, 160)
]
EGA_METAL_MUTANT_PALETTE = [ # Metal Mutant Inspired
    (0, 0, 0), (0, 80, 0), (0, 120, 0), (0, 100, 80),
    (80, 80, 80), (120, 120, 120), (160, 160, 160), (0, 0, 100),
    (50, 50, 120), (80, 200, 80), (80, 200, 200), (160, 80, 40),
    (200, 0, 0), (255, 0, 0), (200, 160, 0), (255, 255, 255)
]
# NEW EGA Palette: Doom-Inspired
EGA_DOOM_PALETTE = [
    (0, 0, 0), (40, 40, 40), (80, 70, 60), (40, 80, 40),
    (120, 0, 0), (80, 0, 80), (100, 60, 40), (100, 100, 100),
    (60, 60, 80), (40, 40, 60), (0, 200, 0), (0, 120, 120),
    (255, 40, 40), (160, 0, 160), (255, 255, 0), (255, 255, 255)
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
    if not os.path.exists("retro_logos_pattern_all"):
        os.makedirs("retro_logos_pattern_all")

    # --- CGA Logos (48x48 split into 2x2 squares of 24x24) ---
    # CGA P0 Normal
    draw_palette_pattern_logo("retro_logos_pattern_all/cga_p0_n_pattern.png", CGA_P0_N_PALETTE, 24, 24)
    # CGA P0 Intense
    draw_palette_pattern_logo("retro_logos_pattern_all/cga_p0_i_pattern.png", CGA_P0_I_PALETTE, 24, 24)
    # CGA P1 Normal
    draw_palette_pattern_logo("retro_logos_pattern_all/cga_p1_n_pattern.png", CGA_P1_N_PALETTE, 24, 24)
    # CGA P1 Intense
    draw_palette_pattern_logo("retro_logos_pattern_all/cga_p1_i_pattern.png", CGA_P1_I_PALETTE, 24, 24)

    # --- EGA Logos (48x48 split into 4x4 squares of 12x12) ---
    # EGA Standard
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_std_pattern.png", EGA_STANDARD_PALETTE, 12, 12)
    # EGA KQ3
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_kq3_pattern.png", PALETTE_KQ3, 12, 12)
    # EGA Thexder
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_thexder_pattern.png", PALETTE_THEXDER, 12, 12)
    # EGA Dune-Inspired
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_dune_pattern.png", EGA_DUNE_PALETTE, 12, 12)
    # EGA Metal Mutant Inspired
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_metal_mutant_pattern.png", EGA_METAL_MUTANT_PALETTE, 12, 12)
    # NEW EGA Doom-Inspired
    draw_palette_pattern_logo("retro_logos_pattern_all/ega_doom_pattern.png", EGA_DOOM_PALETTE, 12, 12)