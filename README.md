Facial emotion recognition project

----------

Files:

Ignore files from [ignore-it] directory. They are irrelevant at this stage.

- data_prep.ipynb

A script for preprocessing KDEF dataset. It works in GoogleColab with GoogleDrive.
The resulting file is data/kdef.npz, which contains 4 np.ndarray objects: x_train, y_train, x_val, y_val
The validation split is set to 0.3.
x_* files contain 100x100 grayscaled, cropped and aligned faces.
y_* files contain one-hot encoded classes as 7D vectors (6 basic emotions + neutral).
Notebook is easily visible on GitHub: https://github.com/PeterRutkowski/fer/blob/master/data_prep.ipynb .

- face_detector.py

Contains a DLIB() class that is used for face detection and alignment.

- camera_stream.py

Contains a CameraStreamDetection() class that is used for live FER.

- main.py

A script for running FER.

- paper.py

The implementation of https://link.springer.com/content/pdf/10.1007/s00521-018-3358-8.pdf .
Notebook version is available here: https://github.com/PeterRutkowski/fer/blob/master/paper.ipynb .
Please see a notebook version, as it has printed outputs.

--------

Current issues solved.