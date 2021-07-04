import cv2
import numpy as np

# Read image
img = cv2.imread("./imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# sobel Filter
K_size = 3

## Zero padding
pad = K_size // 2
out_h = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out_v = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out_h[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out_h.copy()

## Prewitt Filter Horizontal
K_h = [[-1., 0., 1.],[-1., 0., 1.],[-1., 0., 1.]]
## Prewitt Filter Vertical
K_v = [[-1., -1., -1.],[0., 0., 0.], [1., 1., 1.]]

for y in range(H):
    for x in range(W):
        out_h[pad+y, pad+x] = np.mean(K_h * (tmp[y:y+K_size, x:x+K_size]))
        out_v[pad+y, pad+x] = np.mean(K_h * (tmp[y:y+K_size, x:x+K_size]))

out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)
out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out_h)
cv2.imshow("Prewitt Filter Horizontal", out_h)
cv2.imshow("Prewitt Filter Vertical", out_v)
cv2.waitKey(0)
cv2.destroyAllWindows()
