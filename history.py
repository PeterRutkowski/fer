def streamHC(self):
    fd = FaceDetector()
    eternal = True
    while eternal:
        ret, img = self.cam.read()
        fd.detect(img)
        img, detected_faces = fd.face_boundaries()

        for i in range(len(detected_faces)):
            detected_faces[i] = cv2.cvtColor(detected_faces[i], cv2.COLOR_BGR2GRAY)
            detected_faces[i] = cv2.resize(detected_faces[i], (100, 100))
            detected_faces[i] = np.expand_dims(detected_faces[i], axis=0)
            detected_faces[i] = np.expand_dims(detected_faces[i], axis=3)
            detected_faces[i] = np.float16(detected_faces[i])
            print(self.OHE_c[np.argmax(self.classifier.predict(detected_faces[i]))])

        self.__show_img(img)
        self.__show_faces(detected_faces)
