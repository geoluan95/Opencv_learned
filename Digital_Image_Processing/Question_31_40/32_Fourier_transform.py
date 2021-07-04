import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("./imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]


f_img = np.fft.fft2(gray)
    
# Swap the first quadrant and the third quadrant, the second quadrant and the fourth quadrant
f_img =  np.fft.fftshift(f_img)
print(f_img.shape)
# Calculation of power spectrum 
mag = 20*np.log(np.abs(f_img))
    
# Show input image and spectrum image
plt.subplot(121)
plt.imshow(gray, cmap = 'gray')
plt.subplot(122)
plt.imshow(mag, cmap = 'gray')
plt.show()


# DFT
K = W
L = H
M = W
N = H

G = np.zeros((L, K), dtype=np.complex)

x = np.tile(np.arange(W), (H, 1))
y = np.arange(H).repeat(W).reshape(H, -1)

for l in range(L):
    for k in range(K):
        G[l, k] = np.sum(gray * np.exp(-2j * np.pi * (x * k / M + y * l / N))) / np.sqrt(M * N)
        #for n in range(N):
        #    for m in range(M):
        #        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
        #G[l, k] = v / np.sqrt(M * N)

ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
cv2.imwrite("out_ps.jpg", ps)

# IDFT
out = np.zeros((H, W), dtype=np.float32)

for n in range(N):
    for m in range(M):
        out[n,m] = np.abs(np.sum(G * np.exp(2j * np.pi * (x * m / M + y * n / N)))) / np.sqrt(M * N)

out[out>255] = 255
out = out.astype(np.uint8)
    
# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# Destroy all the windows opened before
cv2.destroyAllWindows()

