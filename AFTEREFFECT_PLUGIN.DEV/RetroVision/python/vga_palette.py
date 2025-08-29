import math

# --- Palette Definitions ---

# Standard 16-Color VGA Palette
VGA_PALETTE_16 = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
]

# Function to generate a representative 256-color VGA palette
def generate_vga_palette_256():
    """
    Generates a representative 256-color VGA palette.
    This aims to cover a wide spectrum and includes common ramps.
    """
    palette = []
    
    # --- Indices 0-15: Standard 16 VGA Colors ---
    # These are the most common colors, similar to EGA.
    palette.extend(VGA_PALETTE_16)

    # --- Indices 16-31: Grayscale Ramp ---
    # From black to white, 16 steps.
    for i in range(16):
        gray_val = int((i / 15.0) * 255) # Scale 0-15 to 0-255
        palette.append((gray_val, gray_val, gray_val))

    # --- Indices 32-47: Red Ramp ---
    # From dark red to bright red
    for i in range(16):
        r = int((i / 15.0) * 255) # Red component scales from 0 to 255
        # Ensure R is at least 170 for a noticeable "dark red" start if needed,
        # but for a ramp, starting from 0 and going up is fine.
        # Let's use 170 as a starting point for variety, then scale up.
        r_start = 170
        r = int(r_start + (i / 15.0) * (255 - r_start))
        palette.append((r, 0, 0))

    # --- Indices 48-63: Green Ramp ---
    # From dark green to bright green
    for i in range(16):
        g = int((i / 15.0) * 255)
        palette.append((0, g, 0))
        
    # --- Indices 64-79: Blue Ramp ---
    # From dark blue to bright blue
    for i in range(16):
        b = int((i / 15.0) * 255)
        palette.append((0, 0, b))

    # --- Indices 80-95: Cyan Ramp ---
    # From dark cyan to bright cyan
    for i in range(16):
        c_val = int((i / 15.0) * 255)
        palette.append((0, c_val, c_val))
        
    # --- Indices 96-111: Magenta Ramp ---
    # From dark magenta to bright magenta
    for i in range(16):
        m_val = int((i / 15.0) * 255)
        palette.append((m_val, 0, m_val))

    # --- Indices 112-127: Yellow Ramp ---
    # From dark yellow to bright yellow
    for i in range(16):
        y_val = int((i / 15.0) * 255)
        palette.append((y_val, y_val, 0))

    # --- Indices 128-255: Filling the rest with a mix for variety ---
    # This part is highly variable. For a demo, we can fill with a gradient
    # or repeat some colors. Let's fill with a gradient from a mid-blue to a white.
    # Or simply fill with a mix of shades that might have been used.
    # A common approach is to use a broad range of colors.
    # Let's use a combination: start with more blues, then move to purples, reds, browns etc.
    
    # Let's populate the rest with a more varied set, drawing from the 6-bit DAC steps
    # The VGA DAC has 6 bits per color channel (0-63). We can iterate through these.
    # This loop covers R, G, B combinations.
    color_idx = 0
    for r_bits in range(64): # 6 bits for Red
        for g_bits in range(64): # 6 bits for Green
            for b_bits in range(64): # 6 bits for Blue
                # We only need 256 total colors. If we've already populated up to index 127,
                # we can start filling the rest. This is a simplification.
                # A more accurate generation would map 6-bit R,G,B to 256 indices systematically.
                
                # For this demo, let's just ensure we fill up to 256 colors.
                # We'll use a more direct mapping of color bits for the remaining slots.
                
                # If we have fewer than 256 colors, calculate new ones.
                if len(palette) < 256:
                    # Map 6-bit values (0-63) to 8-bit (0-255). Multiply by ~4.05
                    scaled_r = int(r_bits * (255.0 / 63.0))
                    scaled_g = int(g_bits * (255.0 / 63.0))
                    scaled_b = int(b_bits * (255.0 / 63.0))
                    
                    # To avoid duplicate colors and ensure coverage, we might skip
                    # some combinations or use a different iteration strategy.
                    # A simpler way to fill the remaining slots is to generate a smooth gradient
                    # across the color spectrum.
                    
                    # For this demonstration, let's just add a few more distinct colors
                    # to show the variety.
                    if len(palette) < 128: # Example: adding more blues/cyans/greens
                        palette.append((scaled_r // 2, scaled_g, scaled_b // 2))
                    elif len(palette) < 192: # Example: adding more reds/yellows/browns
                        palette.append((scaled_r, scaled_g // 2, scaled_b // 2))
                    else: # Example: filling the last few with varied colors
                        palette.append((scaled_r, scaled_g, scaled_b))
                
                # Ensure we don't exceed 256 colors
                if len(palette) == 256:
                    break # Exit inner loop
            if len(palette) == 256:
                break # Exit middle loop
        if len(palette) == 256:
            break # Exit outer loop

    # Ensure we have exactly 256 colors. If somehow less, fill the rest with white.
    while len(palette) < 256:
        palette.append((255, 255, 255)) # Fill with white if short

    return palette

# --- Print Palettes ---
def print_palette(name, palette):
    """Prints the palette in a readable format."""
    print(f"--- {name} Palette ({len(palette)} colors) ---")
    for i, color in enumerate(palette):
        # Format as RGB tuple and Hex code
        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
#        print(f"Index {i:03d}: RGB{color}  Hex: {hex_color}")
        print(f"RGB{color}")
    print("-" * (len(name) + 22))
    print("\n") # Add spacing between palettes

if __name__ == "__main__":
    # Print the 16-color VGA palette
    print_palette("Standard 16-Color VGA", VGA_PALETTE_16)

    # Generate and print the representative 256-color VGA palette
    vga_256_palette = generate_vga_palette_256()
    print_palette("Representative 256-Color VGA", vga_256_palette)