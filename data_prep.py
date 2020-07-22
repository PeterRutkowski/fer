import os
import numpy as np
import random

from PIL import Image
from face_detector import DLIB

class PrepareData:
    def __init__(self):
        self.load_path = 'drive/My Drive/project42/KDEF_and_AKDEF/KDEF/'
        self.save_path = 'drive/My Drive/project42/data/'
        self.detector = DLIB()
        self.split = {}
        self.OHE = {'AF': [1, 0, 0, 0, 0, 0, 0], 'AN': [0, 1, 0, 0, 0, 0, 0],
                    'DI': [0, 0, 1, 0, 0, 0, 0], 'HA': [0, 0, 0, 1, 0, 0, 0],
                    'NE': [0, 0, 0, 0, 1, 0, 0], 'SA': [0, 0, 0, 0, 0, 1, 0],
                    'SU': [0, 0, 0, 0, 0, 0, 1], }

    def __split(self, val_ratio):
        folders = np.sort(os.listdir(self.load_path))
        shuffle = random.sample(range(len(folders)), len(folders))
        shuffled_folders = [folders[shuffle[i]] for i in range(len(shuffle))]

        split_border = int((1 - val_ratio) * len(folders))

        self.split['train'] = shuffled_folders[:split_border]
        self.split['val'] = shuffled_folders[split_border:]
        print(self.split)

    def __detect_faces(self):
        for key in self.split.keys():
            x, y = [], []
            for foldername in self.split[key]:
                imagenames = np.sort(os.listdir('{}{}/'.format(self.load_path,
                                                               foldername)))
                for imagename in imagenames:
                    if imagename[6] == 'S':  # straight face
                        img = np.asarray(Image.open('{}{}/{}'.format(self.load_path,
                                                                     foldername, imagename)))
                        self.detector.detect(img)
                        x.append(self.detector.faces[0])
                        y.append(self.OHE[imagename[4:6]])

            x, y = np.asarray(x), np.asarray(y)
            np.savez_compressed('{}kdef_detected_{}.npz'.format(self.save_path, key), x=x, y=y)
            print(x.shape, y.shape)

    def __shuffle(self):
        for key in self.split.keys():
            loaded = np.load('{}kdef_detected_{}.npz'.format(self.save_path, key), allow_pickle=True)
            X, Y = loaded['x'], loaded['y']

            shuffle = random.sample(range(len(X)), len(X))

            x = [X[shuffle[i]] for i in range(len(shuffle))]
            y = [Y[shuffle[i]] for i in range(len(shuffle))]

            x, y = np.asarray(x), np.asarray(y)
            np.savez('{}kdef_shuffled_{}.npz'.format(self.save_path, key), x=x, y=y)
            print(x.shape, y.shape)

    def __save(self, val_split):
        loaded_train = np.load('{}kdef_detected_train.npz'.format(self.save_path), allow_pickle=True)
        x_train, y_train = loaded_train['x'], loaded_train['y']
        loaded_val = np.load('{}kdef_detected_val.npz'.format(self.save_path), allow_pickle=True)
        x_val, y_val = loaded_val['x'], loaded_val['y']
        np.savez('{}kdef_{}.npz'.format(self.save_path, val_split),
                 x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    def prepare(self, val_split):
        self.__split(val_split)
        print('Split done...')
        self.__detect_faces()
        print('Face detection done...')
        self.__shuffle()
        print('Shuffle done...')
        self.__save(val_split)
        print('Data saved.')

pd = PrepareData()
pd.prepare(0.3)