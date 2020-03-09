"""
Script para seleccionar manualmente los registros de farderos almacenados
previamente
"""

import io
import os
import csv
import cv2
import base64
import pickle
import numpy as np

from FaceDetector import FaceDetectorDlib
from FaceAlign import get_aligned_face, get_landmarks_visualization

def main():
    """
    Se extrae la información de un csv
    """
    registros_data = []
    path = 'database/registros.csv'
    with io.open(path, newline = '', mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            registros_data.append(row)

    detector = FaceDetectorDlib()

    filtered_registros = []
    for reg in registros_data:
        idrg = reg['id_registro_persona']
        name = reg['nombre']
        im64 = reg['foto']
        comm = reg['comentarios']
        stor = reg['id_tienda']
        idgf = reg['id_globalface']

        if stor == '58':
            continue
        try:
            im64dec = base64.standard_b64decode(im64)
        except:
            pass
            continue

        imar = np.fromstring(base64.standard_b64decode(im64), np.uint8)
        img = cv2.imdecode(imar, cv2.IMREAD_COLOR)

        """
        Se extraen rostros de las imágenes guardadas en csv
        """
        box = detector.detect(img)
        if len(box) is 0:
            continue
        else:
            box = box[0]

        """
        Se alinean los rostros y se muestran los puntos faciales para comprobar
        si es un rostro útil
        """
        crop = get_aligned_face(img, box)
        lands = get_landmarks_visualization(img, box)

        cv2.imshow('img', img)
        cv2.imshow('crop', crop)
        cv2.imshow('lands', lands)
        print("{:<6s}{:<50s}".format(idgf, name))

        key = cv2.waitKey(0)
        keypress = str(chr(key & 0xFF))


        """
        Presionar 'y' para aceptar el rostro, 'n' para rechazar, 'q' para salir
        """
        if keypress == 'q':
            break
        elif keypress == 'y':
            print("Yes")
            filtered_registros.append(reg)
            # print(reg)
        elif keypress == 'n':
            print("No")

    # pickle.dump(filtered_registros, open('registros/registros_filtrados.pkl', 'wb'))
    # print("Saved registros")

    """
    Los rostros aceptados se guardan en 'database/registros_filtrados.csv' para
    agregarse en la base de datos posteriormente
    """
    csvname = 'database/registros_filtrados.csv'
    with open(csvname, 'w', newline = '') as csvfile:
        fieldnames = ['id_registro_persona',
                      'nombre',
                      'foto',
                      'comentarios',
                      'id_tienda',
                      'id_globalface']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for filt in filtered_registros:
            fil = dict(filt)
            writer.writerow(fil)



if __name__ == "__main__":
    main()
