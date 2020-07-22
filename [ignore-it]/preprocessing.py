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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img)

    def print(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#p = Preprocessor()
#img = cv2.imread('trump2.jpg')
#img = p.GaussianSmoothing(img)
#img = p.BilateralFilterSmoothing(img)
#img = p.HistogramEqualisation(img)
#p.print(img)
