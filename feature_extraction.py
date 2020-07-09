import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class LocalBinaryPatterns:
    def __init__(self):
        print()

    def __padding_comparison(self,img, i, j, x, y):
        if img[x][y] > img[i][j]:
            return 1
        else:
            return 0

    def __binary_padding(self, img, i, j):
        # anticlockwise; start in upper left corner
        padding = []
        padding.append(self.__padding_comparison(img, i, j, i-1, j-1))
        padding.append(self.__padding_comparison(img, i, j, i-1, j))
        padding.append(self.__padding_comparison(img, i, j, i-1, j+1))
        padding.append(self.__padding_comparison(img, i, j, i, j+1))
        padding.append(self.__padding_comparison(img, i, j, i+1, j+1))
        padding.append(self.__padding_comparison(img, i, j, i+1, j))
        padding.append(self.__padding_comparison(img, i, j, i+1, j-1))
        padding.append(self.__padding_comparison(img, i, j, i, j-1))
        return padding

    def convert(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = np.zeros(img.shape)
        height, width = img.shape[0], img.shape[1]
        img = np.pad(img, ((1, 1), (1, 1)), 'constant')
        for i in range(1,height+1):
            for j in range(1,width+1):
                binary_padding = self.__binary_padding(img, i, j)
                bin = 0
                powers = [128, 64, 32, 16, 8, 4, 2, 1]
                for k in range(len(binary_padding)):
                    bin += binary_padding[k] * powers[k]
                template[i-1,j-1] = bin
        template = np.uint8(template)
        return template

class FeatureExtractor:
    def __init__(self):
        print()
        self.LBP = LocalBinaryPatterns()

    def save_image(self, img):
        img = Image.fromarray(img, 'L')
        img.save('trial.png')

    def extract_features(self):
        print()

img = cv2.imread('test.jpeg')
extractor = FeatureExtractor()
img = extractor.LBP.convert(img)
extractor.save_image(img)