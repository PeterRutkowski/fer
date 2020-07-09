from face_detector import FaceDetector
from preprocessing import Preprocessor
import cv2

class FER:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.preprocessor = Preprocessor()

    def __show_face(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def trial(self, img):
        self.face_detector.detect(img)
        faces = self.face_detector.extract_faces()

        for face in faces:
            self.__show_face(face)
            #self.__show_face(self.preprocessor.BilateralFilterSmoothing(face))
            self.__show_face(self.preprocessor.HistogramEqualisation(face))

img = cv2.imread('trump.jpeg')
model = FER()
model.trial(img)