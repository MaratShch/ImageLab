import numpy as np
from PIL import Image

h, w = 720, 1280
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

# horizontal luminance ramp, vertical hue sweep, plus blown highlight discs
img = np.zeros((h, w, 3), dtype=np.float32)
ramp = xx / (w - 1)
img[:, :, 0] = ramp
img[:, :, 1] = ramp
img[:, :, 2] = ramp

# colour patches
for i, col in enumerate([(1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1)]):
    y0 = 40 + i*20
    img[y0:y0+18, 60:1220] = np.array(col, dtype=np.float32) * ramp[y0:y0+18, 60:1220, None]

# hard specular highlights to exercise halation
for cx, cy, r in [(300, 500, 26), (640, 520, 14), (980, 480, 40)]:
    m = ((xx-cx)**2 + (yy-cy)**2) < r*r
    img[m] = 1.0

# mid grey patch for round-trip check
img[600:680, 40:160] = 0.4663  # sRGB encoding of 0.18 linear

# fine detail bars to exercise MTF
for k, period in enumerate([4, 8, 16, 32]):
    y0 = 300 + k*24
    bars = ((xx // (period/2)) % 2).astype(np.float32)
    img[y0:y0+20, 700:1240] = bars[y0:y0+20, 700:1240, None]*0.8 + 0.1

Image.fromarray((np.clip(img,0,1)*255+0.5).astype(np.uint8)).save("test_chart.png")
print("wrote test_chart.png")
