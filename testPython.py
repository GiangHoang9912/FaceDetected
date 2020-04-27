import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('C:\\Users\\giang\\Desktop\\TestVScode\\Images\\digits.png', 0)

# cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]

# x = np.array(cells)

# train = x[:, 0:100].reshape(-1, 400).astype(np.float32)

# testDigits = cv2.imread('C:\\Users\\giang\\Desktop\\TestVScode\\Images\\1.jpg', 0).reshape(-1, 400).astype(np.float32)

# k = np.arange(10)

# train_label = np.repeat(k, 250)[:, np.newaxis]

# knn = cv2.ml.KNearest_create()
# knn.train(train, 0, train_label)

# rs1, rs2, rs3, rs4 = knn.findNearest(testDigits, 5)

# print("this picture is number : "+format(rs2[0][0]))

import os

arr = os.listdir("C:\\Users\\giang\\Desktop\\TestVScode\\Anh Noo")

trainData = []

for i in range(6):
    img = cv2.imread("C:\\Users\\giang\\Desktop\\TestVScode\\Anh Noo\\"+ arr[i], 0)
    img = cv2.resize(img, (200, 200))

    height = img.shape[0]
    width = img.shape[1]

    print(height)
    print(width)
    trainData.append(img.reshape(-1, 4000).astype(np.float32))

imgTestCase = cv2.imread("C:\\Users\\giang\\Desktop\\TestVScode\\Anh Noo\\testcase.jpg", 0)
imgTestCase = cv2.resize(imgTestCase, (200, 200))


testcase = imgTestCase.reshape(-1, 4000).astype(np.float32)

knn = cv2.ml.KNearest_create()

k = np.arange(5)

train_label = np.repeat(k, 1)[:, np.newaxis]

knn.train(trainData,0 , train_label)

result = knn.findNearest(testcase, 5)

print(result)



