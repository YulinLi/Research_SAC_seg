import cv2
import numpy as np

height = 128
width = 128

x1, y1 = 25, 25
x2, y2 = 75, 100

img = np.zeros((height, width, 3), np.uint8)
img.fill(255)  # white

img[x1:x2, y1:y2, :] = (255, 0, 0)

cv2.imwrite("image/example.png", img)
