from PIL import Image
import math

# --- Parameters ---
width, height = 48, 48
steps = 16  # number of intensity levels
color_type = "green"  # "green", "amber", or "white"
gamma = 2.2  # gamma correction for CRT-like brightness response

# --- Function: get RGB value for given intensity ---
def hercules_color(level, color_type="green"):
    intensity = (level / (steps - 1)) ** (1 / gamma)
    if color_type == "green":
        # Classic Hercules green phosphor (slightly yellowish at high intensity)
        r = int(60 * intensity)
        g = int(255 * intensity)
        b = int(60 * intensity / 2)
    elif color_type == "amber":
        # Amber phosphor (warm golden tone)
        r = int(255 * intensity)
        g = int(160 * intensity)
        b = int(40 * intensity / 2)
    elif color_type == "white":
        # Monochrome grayscale
        val = int(255 * intensity)
        r = g = b = val
    else:
        raise ValueError("Unsupported color type")
    return (r, g, b, 255)

# --- Create RGBA image ---
img = Image.new("RGBA", (width, height))

# --- Draw horizontal stripes for each intensity level ---
stripe_height = height // steps
for i in range(steps):
    color = hercules_color(i, color_type)
    y0 = i * stripe_height
    for y in range(y0, y0 + stripe_height):
        for x in range(width):
            img.putpixel((x, y), color)

# --- Save output ---
filename = f"hercules_palette_{color_type}_48x48.png"
img.save(filename, "PNG")
print(f"Saved: {filename}")
