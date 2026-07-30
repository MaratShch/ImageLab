import numpy as np
from PIL import Image, ImageDraw

W, H = 3200, 1800
img = np.zeros((H, W, 3), dtype=np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
img[:] = (xx / (W - 1))[:, :, None] * 0.85 + 0.05

cols = [(0.75,0.10,0.10),(0.10,0.60,0.15),(0.10,0.20,0.75),(0.80,0.70,0.15),
        (0.85,0.55,0.45),(0.55,0.30,0.55)]
bw = W // len(cols)
for i, c in enumerate(cols):
    img[120:520, i*bw+40:(i+1)*bw-40] = c

for cx, cy, r in [(700,1200,90),(1600,1250,40),(2500,1150,140)]:
    img[((xx-cx)**2 + (yy-cy)**2) < r*r] = 1.0

for k, period in enumerate([6, 12, 24, 48]):
    y0 = 700 + k*60
    bars = ((xx // (period/2)) % 2).astype(np.float32)
    img[y0:y0+48, 1800:3100] = bars[y0:y0+48, 1800:3100, None]*0.75 + 0.10

img[1500:1700, 100:400] = 0.4663
Image.fromarray((np.clip(img,0,1)*255+0.5).astype(np.uint8)).save("period_chart.png")
print("period_chart.png", W, H)
