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

# Make the labels for each digit 0-9
k = np.arange(10)
train_label = np.repeat(k, 250)[:,np.newaxis]
test_label = train_label.copy()

knn = cv.ml.KNearest.create()
knn.train(train, cv.ml.ROW_SAMPLE, train_label)
ret, results, neighbours, dist = knn.findNearest(test, k = 5)

matches = results == test_label
correct = np.count_nonzero(matches)
accuracy = correct / results.size * 100.0
print(accuracy)