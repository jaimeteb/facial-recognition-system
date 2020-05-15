"""
Detectores de rostros.
"""

import cv2
from numpy import clip

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
