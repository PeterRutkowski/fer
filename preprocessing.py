import cv2
import numpy as np

def print_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Preprocessing:
    def __init__(self):
        print()

    def GaussianSmoothing(self, img):
        return cv2.GaussianBlur(img, (5,5), 0)

    def BilateralFilterSmoothing(self, img):
        return cv2.bilateralFilter(img,15,75,75)

img = cv2.imread('trump2.jpg')
p = Preprocessing()
#img = p.GaussianSmoothing(img)
img = p.BilateralFilterSmoothing(img)

print_img(img)


