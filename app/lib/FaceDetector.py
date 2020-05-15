"""
Detectores de rostros.
"""

import cv2
import dlib
from numpy import clip

swaprb = lambda x: x[...,::-1]

class FaceDetectorDlib:
    """
    Detector de rostros de la librería Dlib. Es el detector más preciso pero
    el más pesado.
    """
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        image = swaprb(image)
        dets = self.detector(image, 1)
        return [((d.left(), d.top()), (d.right(), d.bottom())) for d in dets]

class FaceDetectorOpenVino:
    """
    Detector de rostros del OpenVino Toolkit de Intel. Es el detector más
    eficiente pero no funciona sin las variables de entorno de OpenVino.
    """
    def __init__(self):
        self.net = cv2.dnn.readNet('lib/models/face-detection-retail-0004.xml',
                                   'lib/models/face-detection-retail-0004.bin')

    def detect(self, image):
        h, w, _ = image.shape

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor = 1,
            size = (300, 300),
            mean = (0, 0, 0),
            swapRB = False,
            crop = False
        )

        self.net.setInput(blob)
        output = self.net.forward()[0][0]
        return [((int(det[3] * w), int(det[4] * h)), (int(det[5] * w), int(det[6] * h))) for det in output if det[2] > 0.8]


class FaceDetectorHaar:
    """
    Detector de rostros con HaarCascade. Es el más rápido pero el menos preciso.
    Se amplía la detección por un factor "amp" para mejorar el espacio de
    detección.
    """
    def __init__(self, amp=0.35):
        self.face_cascade = cv2.CascadeClassifier('lib/models/haarcascade_frontalface_default.xml')
        self.amp = amp

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imh, imw = gray.shape
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        clips = [(clip(x-int(w*self.amp),0,imw), clip(y-int(h*self.amp),0,imh), clip(x+w+int(w*self.amp),0,imw), clip(y+h+int(h*self.amp),0,imh)) for (x,y,w,h) in faces]
        return [(((l,t), (r,b)), (slice(t,b), slice(l,r))) for (l,t,r,b) in clips]
