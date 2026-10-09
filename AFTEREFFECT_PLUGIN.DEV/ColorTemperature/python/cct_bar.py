import colour
import numpy as np

print("constexpr std::array<uint8_t, 240 * 3> CCT_DEBUG_TABLE = {")

for i in range(240):
    cct = 1000 + (i * 100)
    
    # 1. Ohno 2013 calculates strictly in CIE 1960 UCS space -> outputs (u, v)
    uv = colour.temperature.CCT_to_uv(np.array([cct, 0.0]), method='Ohno 2013')
    
    # 2. Translate 1960 (u, v) to 1931 (x, y)
    xy = colour.UCS_uv_to_xy(uv)
    
    # 3. Convert (x, y) to CIE XYZ
    XYZ = colour.xy_to_XYZ(xy)
    
    # 4. Convert XYZ to sRGB (Applies standard IEC linearization curves)
    RGB = colour.XYZ_to_sRGB(XYZ)
    
    # 5. Scale to 8-bit and clamp strictly to bounds
    RGB = np.clip(RGB * 255.0, 0, 255).astype(int)
    
    # Format as C++14 array elements
    print(f"    {RGB[0]:3d}, {RGB[1]:3d}, {RGB[2]:3d}, // [{i:3d}] {cct}K")

print("};")