import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        print()

    def GaussianSmoothing(self, img):
        return cv2.GaussianBlur(img, (5,5), 0)

    def BilateralFilterSmoothing(self, img):
        return cv2.bilateralFilter(img,15,75,75)

    def HistogramEqualisation(self, img):
        return cv2.equalizeHist(img)

