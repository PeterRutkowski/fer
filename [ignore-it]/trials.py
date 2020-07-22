from face_detector import FaceDetector
from preprocessing import Preprocessor
from feature_extraction import FeatureExtractor
import cv2

class FER:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.preprocessor = Preprocessor()
        self.extractor = FeatureExtractor()

    def __show_face(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def trial(self, img):
        # detection
        #self.face_detector.detect(img, scaleFactor=1.03, minNeghbors=5)
        #self.face_detector.print_with_boundaries()

        # preprocessing images
        #self.__show_face(img)
        #self.__show_face(self.preprocessor.GaussianSmoothing(img))
        #self.__show_face(self.preprocessor.BilateralFilterSmoothing(img))
        #self.__show_face(self.preprocessor.HistogramEqualisation(img))

        # preprocessing faces
        #faces = self.face_detector.extract_faces()
        #for face in faces:
        #    self.__show_face(face)
        #    self.__show_face(self.preprocessor.GaussianSmoothing(face))
        #    self.__show_face(self.preprocessor.BilateralFilterSmoothing(face))
        #    self.__show_face(self.preprocessor.HistogramEqualisation(face))



img = cv2.imread('trump.jpeg')
#img = cv2.imread('trump2.jpg')
#img = cv2.imread('test.jpeg')
model = FER()
model.trial(img)