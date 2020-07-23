import os
import numpy as np
import cv2
from PIL import Image
from face_detector import DLIB

import tensorflow as tf
from tensorflow.keras.models import load_model

class Tester:
    def __init__(self):
        self.detector = DLIB()
        self.classifier = load_model('models/cnn_batch32_adam_91.837.h5')
        self.OHE = {'AF': [1, 0, 0, 0, 0, 0, 0], 'AN': [0, 1, 0, 0, 0, 0, 0],
                    'DI': [0, 0, 1, 0, 0, 0, 0], 'HA': [0, 0, 0, 1, 0, 0, 0],
                    'NE': [0, 0, 0, 0, 1, 0, 0], 'SA': [0, 0, 0, 0, 0, 1, 0],
                    'SU': [0, 0, 0, 0, 0, 0, 1], }
        self.OHE_c = {0: 'AFRAID', 1: 'ANGRY', 2: 'DISGUSTED', 3: 'HAPPY',
                      4: 'NEUTRAL', 5: 'SAD', 6: 'SURPRISED'}
        self.y = np.empty(0)
        self.prediction = np.empty(0)
        self.imagenames = np.empty(0)

    def process_images(self):
        imagenames = np.sort(os.listdir('faces/'))[1:]  # on macOS ignore .DS_Store file
        x, y = [], []
        for imagename in imagenames:
            img = np.uint8(Image.open('faces/{}'.format(imagename)))
            self.detector.detect(img)
            # print(imagename, len(self.detector.faces))
            x.append(self.detector.faces[0])
            y.append(self.OHE[imagename[0:2]])

        x, y, imagenames = np.asarray(x), np.asarray(y), np.asarray(imagenames)
        np.savez_compressed('data/test.npz', x=x, y=y, imagenames=imagenames)
        print(x.shape, y.shape)

    def test(self):
        loaded = np.load('data/test.npz')
        x, self.y, self.imagenames = loaded['x'], loaded['y'], loaded['imagenames']

        x = np.expand_dims(x, axis=3)
        x = tf.convert_to_tensor(x)
        self.prediction = np.round(self.classifier(x), 2)

    def eval_accuracy(self):
        total_counter = 0
        for i in range(len(self.OHE_c.keys())):
            class_counter = 0
            for j in range(i*10,(i+1)*10):
                if np.argmax(self.y[j]) == np.argmax(self.prediction[j]):
                    total_counter += 1
                    class_counter += 1
            print('Accuracy {}: {}'.format(self.OHE_c[i], class_counter/10))
        print('Accuracy TOTAL: {}'.format(total_counter/70))

    def eval_scores(self):
        for i in range(len(self.y)):
            print(self.imagenames[i])
            print(self.y[i], self.OHE_c[np.argmax(self.y[i])])
            print(self.prediction[i], self.OHE_c[np.argmax(self.prediction[i])])

t = Tester()
#t.process_images()
t.test()
t.eval_accuracy()