import numpy as np
import cv2 as cv

img = cv.imread('digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cells = []
for row in np.vsplit(gray, 50):
    cells.append(np.hsplit(row, 100))

x = np.array(cells)

train = x[:,:50].reshape(-1, 400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)
