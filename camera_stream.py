from face_detector import DLIB
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2

class CameraStreamDetection:
    def __init__(self):
        self.dlib = DLIB() # face detector
        self.classifier = load_model('models/cnn_batch32_adam_91.837.h5')
        self.cam = cv2.VideoCapture(0) # init camera
        self.OHE = {'AF': [1, 0, 0, 0, 0, 0, 0], 'AN': [0, 1, 0, 0, 0, 0, 0], 'DI': [0, 0, 1, 0, 0, 0, 0],
                    'HA': [0, 0, 0, 1, 0, 0, 0], 'NE': [0, 0, 0, 0, 1, 0, 0], 'SA': [0, 0, 0, 0, 0, 1, 0],
                    'SU': [0, 0, 0, 0, 0, 0, 1], } # classes one hot encoding
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
        eternity = True
        while eternity:
            ret, img = self.cam.read() # capture image
            img = imutils.resize(img, width=800) # resize image
            self.dlib.detect(img) # detect faces
            detected_faces = self.dlib.faces

            for face in detected_faces: # for each face use classifier
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=3)
                face = tf.convert_to_tensor(face)
                prediction = np.asarray(self.classifier.predict(face))
                print(np.round(prediction,2), self.OHE_c[np.argmax(prediction)])
            self.__show_img(img) # print camera capture
            self.__show_faces(detected_faces) # print detected faces

        self.cam.release()
        cv2.destroyAllWindows()