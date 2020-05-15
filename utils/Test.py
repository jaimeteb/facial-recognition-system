"""
Script para probar el sistema de reconocimiento facial con una cámara IP
"""

import os
import cv2
import json
import base64
import requests

from datetime import datetime
from lib.FaceDetectorHaar import FaceDetectorHaar

# video_path = 'http://root:Gb659w&3NrNy@172.16.3.46/axis-cgi/mjpg/video.cgi'
video_path = 0
cap = cv2.VideoCapture(video_path)
face_detector = FaceDetectorHaar()
detections_path = 'x'

def post(data):
    """
    Función para realizar peticiones a la API de reconocimiento facial
    """
    try:
        r = requests.post("http://localhost:8000/api/features", data=json.dumps(data))
        resp = json.loads(r.text)
        return resp
    except ConnectionRefusedError:
        return None


while True:
    ok, frame = cap.read()
    if ok:
        preview = frame.copy()
        """
        Se realiza detección de rostros con HaarCascade sobre el frame
        """
        boxes = face_detector.detect(frame)
        for box, (sy, sx) in boxes:
            cv2.rectangle(preview, box[0], box[1], (0, 255, 0), 1)
            crop = frame[sy, sx, :]

            """
            Se codifica en base64 cada rostro encontrado y se envía a la API
            """
            _, enccrop = cv2.imencode('.jpg', crop)
            b64crop = base64.b64encode(enccrop)
            b64crop = b64crop.decode("utf-8")
            data = {"imagen": b64crop}

            id = post(data)
            if id is not None:
                print(id)
                if id["pred"] is not None:
                    """
                    Si hay una identificación de rostro, se guarda la imagen y
                    se muestran los resultados
                    """
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    pic = os.path.join(detections_path, "{} - {} - {}.jpg".format(timestamp, id["pred"], id["sim"]))
                    cv2.imwrite(pic, preview)


        cv2.imshow("preview", preview)

        k = cv2.waitKey(33)
        if k == 27:    # Esc key to stop
            break
