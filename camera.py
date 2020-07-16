from face_detector import DLIB
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2

class CameraStreamDetection:
    def __init__(self):
        self.dlib = DLIB()
        self.classifier = load_model('cnn.h5')
        self.cam = cv2.VideoCapture(0)
        self.OHE = {'AF': [1, 0, 0, 0, 0, 0, 0], 'AN': [0, 1, 0, 0, 0, 0, 0], 'DI': [0, 0, 1, 0, 0, 0, 0],
                    'HA': [0, 0, 0, 1, 0, 0, 0], 'NE': [0, 0, 0, 0, 1, 0, 0], 'SA': [0, 0, 0, 0, 0, 1, 0],
                    'SU': [0, 0, 0, 0, 0, 0, 1], }
        self.OHE_c = {0: 'AFRAID', 1: 'ANGRY', 2: 'DISGUSTED', 3: 'HAPPY',
                      4: 'NEUTRAL', 5: 'SAD', 6: 'SURPRISED'}

    def __show_img(self, img):
        cv2.imshow('img', np.uint8(img))
        cv2.waitKey(1)
    def __show_faces(self, faces):
        for i in range(len(faces)):
            cv2.imshow('{}'.format(i), np.uint8(faces[i]))
            cv2.waitKey(1)

    def streamDLIB(self):
        eternal = True
        while eternal:
            ret, img = self.cam.read()
            img = imutils.resize(img, width=700)
            self.dlib.detect(img)
            detected_faces = self.dlib.faces
            print(detected_faces.shape)
            for face in detected_faces:
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=3)
                print(self.OHE_c[np.argmax(self.classifier.predict(face))])
            self.__show_img(img)
            self.__show_faces(detected_faces)

        self.cam.release()
        cv2.destroyAllWindows()