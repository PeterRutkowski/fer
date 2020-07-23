import cv2
import numpy as np
import dlib
from imutils.face_utils import FaceAligner

class DLIB:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.FaceAligner = FaceAligner(self.predictor, desiredFaceWidth=100, desiredLeftEye=(0.22,0.22))
        self.faces = np.empty(0)

    def detect(self, img):
        # return a np.ndarray containing datected grayscaled faces
        faces = []
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
        rects = self.detector(gray, 2) # calculate rectangles of detected faces
        for rect in rects:
            if img.ndim == 2:
                faces.append(self.FaceAligner.align(img, gray, rect))
            else:
                faces.append(cv2.cvtColor(self.FaceAligner.align(img, gray, rect), cv2.COLOR_BGR2GRAY))
        self.faces = np.float16(faces)