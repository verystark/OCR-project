import numpy as np
import cv2 as cv
import tkinter as tk
from sklearn.utils import shuffle

root = tk.Tk()
myCanvas = tk.Canvas(root, bg='white', height=400, width=400, bd=0, cursor='circle')
img = cv.imread('digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cells = []
for row in np.vsplit(gray, 50):
    cells.append(np.hsplit(row, 100))

x = np.array(cells).reshape(-1, 400).astype(np.float32)

labels = np.repeat(np.arange(10), 500)[:, np.newaxis]
x, labels = shuffle(x, labels, random_state=42)

train = x[:4000]
test = x[4000:]
train_label = labels[:4000]
test_label = labels[4000:]

knn = cv.ml.KNearest.create()
knn.train(train, cv.ml.ROW_SAMPLE, train_label)
ret, results, neighbours, dist = knn.findNearest(test, k = 5)

matches = results == test_label
correct = np.count_nonzero(matches)
accuracy = correct / results.size * 100.0
print(accuracy)

