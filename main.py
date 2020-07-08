import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        print('Face detector created!')

    def __face_boundaries(self, img, faces):
        for rectangle in faces:
            x, y, w, h = rectangle
            img = cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 3)

    def __greyscale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def HaarCascade(self):
        HC = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        img = cv2.imread('trump.jpeg')
        self.__greyscale(img)

        faces = HC.detectMultiScale(img, 1.03, 6)  # detect faces

        print(faces)

        self.__face_boundaries(img, faces)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



det = FaceDetector()
det.HaarCascade()