import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.__HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.faces = np.empty(0)
        self.img = np.empty(0)

    def print_with_boundaries(self):
        assert self.faces.size, 'DetectorAssert: no faces @ print_face_boundaries'
        assert self.img.size, 'DetectorAssert : no image @ print_face_boundaries'
        img = self.img
        for rectangle in self.faces:
            x, y, w, h = rectangle
            img = cv2.rectangle(self.img, (x, y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __greyscale(self):
        assert self.img.size, 'DetectorAssert: no image @ __greyscale'
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect(self, img=None, scaleFactor=1.03, minNeghbors=6):
        if img is not None:
            self.img = img

        assert self.img.size, 'DetectorAssert : no image @ detect'
        detector = self.__HaarCascade
        self.__greyscale()
        self.faces = detector.detectMultiScale(self.img, scaleFactor, minNeghbors)