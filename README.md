Facial emotion recognition project

----------

Files:

Ignore files from [ignore-it] directory. They are irrelevant at this stage.

- data_prep.py

A class for preprocessing KDEF dataset stored in GoogleDrive.
The resulting file contains 4 np.ndarray objects: x_train, y_train, x_val, y_val.
The validation split is a parameter.
x_* files contain 100x100 grayscaled, cropped and aligned faces.
y_* files contain one-hot encoded classes as 7D vectors (6 basic emotions + neutral).

- face_detector.py

Contains a DLIB() class that is used for face detection and alignment.

- camera_stream.py

Contains a CameraStreamDetection() class that is used for live FER.

- main.py

A script for running FER.

- paper.ipynb

The implementation of https://link.springer.com/content/pdf/10.1007/s00521-018-3358-8.pdf .
The notebook is also available here: https://github.com/PeterRutkowski/fer/blob/master/paper.ipynb .

--------

Current issues solved.
