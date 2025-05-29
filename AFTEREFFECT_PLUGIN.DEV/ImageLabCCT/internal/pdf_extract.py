# Re-import required modules after code execution state reset
from pathlib import Path
import fitz  # 
import re

# Reload the file
pdf_path = Path("Blackbody_color_datafile.pdf")
text_path = Path("CCT_RGB_1931_1000K_25000K_step100K.txt")

# Read the PDF content
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()

# Regex to match rows with CIE 1931 2° observer data and extract RGB
pattern = re.compile(r"""
    ^\s*                # start of line
    (\d{4})\s+K\s+      # CCT in Kelvin
    2deg\s+             # CIE 1931 2° observer
    [\d.]+\s+[\d.]+\s+  # x y (skip)
    [\de+.-]+\s+        # power
    ([\d.]+)\s+         # R normalized
    ([\d.]+)\s+         # G normalized
    ([\d.]+)\s+         # B normalized
    (\d+)\s+(\d+)\s+(\d+)  # R G B integer
""", re.MULTILINE | re.VERBOSE)

# Build the output file content
lines = ["// R  G  B\t\t\t CCT"]
for match in pattern.finditer(text):
    cct = int(match.group(1))
    if 1000 <= cct <= 9900 and cct % 100 == 0:
        print(f"Process CCT line {cct}") 
        r = int(match.group(5))
        g = int(match.group(6))
        b = int(match.group(7))
        lines.append(f"ColorTriplet{{{r}, {g}, {b}}},\t\t// {cct} Kelvin Degrees")

pattern = re.compile(r"""
    ^\s*                # start of line
    (\d{5})\s+K\s+      # CCT in Kelvin
    2deg\s+             # CIE 1931 2° observer
    [\d.]+\s+[\d.]+\s+  # x y (skip)
    [\de+.-]+\s+        # power
    ([\d.]+)\s+         # R normalized
    ([\d.]+)\s+         # G normalized
    ([\d.]+)\s+         # B normalized
    (\d+)\s+(\d+)\s+(\d+)  # R G B integer
""", re.MULTILINE | re.VERBOSE)
for match in pattern.finditer(text):
    cct = int(match.group(1))
    if 10000 <= cct <= 25000 and cct % 100 == 0:
        print(f"Process CCT line {cct}") 
        r = int(match.group(5))
        g = int(match.group(6))
        b = int(match.group(7))
        lines.append(f"ColorTriplet{{{r}, {g}, {b}}},\t\t// {cct} Kelvin Degrees")


# Save to file
text_path.write_text("\n".join(lines))

text_path.name  # Return filename for download

