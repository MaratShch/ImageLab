"""Contact sheet of every stock, for eyeballing. Not part of the test suite."""
from pathlib import Path
from PIL import Image, ImageDraw
from film_profiles import FILM_PROFILES

files = [(p, Path("film_renders") / f"test_chart_{p.name}.png") for p in FILM_PROFILES]
files = [(p, f) for p, f in files if f.exists()]
tw, th, cols = 300, 169, 3
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 15)), (20, 20, 20))
d = ImageDraw.Draw(sheet)
for i, (p, f) in enumerate(files):
    with Image.open(f) as im:
        im = im.convert("RGB").resize((tw, th), Image.LANCZOS)
    x, y = (i % cols) * tw, (i // cols) * (th + 15)
    sheet.paste(im, (x, y + 15))
    if p.has_reseau:
        tag = "mosaic"
    elif p.is_reversal:
        tag = "rev"
    elif p.is_monochrome:
        tag = "bw"
    else:
        tag = "neg"
    d.text((x + 3, y + 3), f"{p.name}  [{tag}]", fill=(225, 225, 225))
sheet.save("contact_sheet.png")
print("contact_sheet.png", sheet.size)
