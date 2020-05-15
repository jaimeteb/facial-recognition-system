"""
Script para inicializar la base de datos con los rostros de farderos manualmente
selccionados
"""

import io
import os
import csv
import cv2
import base64
import psycopg2
import numpy as np

from random import sample
from lib.Augmentate import seq

from lib.FaceDetector import FaceDetectorDlib
from lib.FaceNetKeras import get_features_tolist
from lib.FaceAlign import get_aligned_face


def main():
    """
    Conexión a la base de datos
    """
    con = psycopg2.connect(
        dbname = os.environ["POSTGRES_DB"],
        user = os.environ["POSTGRES_USER"],
        password = os.environ["POSTGRES_PASSWORD"],
        host = os.environ["POSTGRES_HOST"],
        port = os.environ["POSTGRES_PORT"]
    )
    con.autocommit = True
    cur = con.cursor()

    """
    Cargar registros guardados
    """
    path = 'database/registros_filtrados.csv'
    registros_filtrados = []
    with io.open(path, newline = '', mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            registros_filtrados.append(row)

    face_detector = FaceDetectorDlib()

    for reg in registros_filtrados:
        idrg = reg['id_registro_persona']
        name = reg['nombre']
        im64 = reg['foto']
        comm = reg['comentarios']
        stor = reg['id_tienda']
        idgf = reg['id_globalface']

        imar = np.fromstring(base64.standard_b64decode(im64), np.uint8)
        img = cv2.imdecode(imar, cv2.IMREAD_COLOR)

        box = face_detector.detect(img)
        if len(box) is 0:
            continue
        else:
            box = box[0]
        crop = get_aligned_face(img, box)

        augments = ["false"]
        ims_src = [im64]
        feats = [get_features_tolist(crop)]

        """
        Agregar información a la tabla registros
        """
        execstr = "select * from put_registros({}, '{}', '{}', {}, {});".format(idgf, name, comm, stor, 2)
        cur.execute(execstr)
        idrg = cur.fetchall()[0][0]

        """
        Realizar aumento de información hasta tener 10 imágenes por registro
        """
        while len(ims_src) < 10:
            augim = seq.augment_images([img])[0]
            box = face_detector.detect(augim)
            if len(box) is not 0:
                box = box[0]
                crop = get_aligned_face(augim, box)
                feats.append(get_features_tolist(crop))

                _, enccrop = cv2.imencode('.jpg', crop)
                b64crop = base64.b64encode(enccrop)
                ims_src.append(b64crop.decode("utf-8"))
                augments.append("true")

        """
        Agregar features de cada imagen a la tabla features
        """
        for src, fts, aug in zip(ims_src, feats, augments):
            execstr = "select * from put_features({}, {}, '{}', array{}, {});".format(idrg, idgf, src, fts, aug)

            cur.execute(execstr)
            cur.fetchall()

    con.close()




if __name__ == "__main__":
    main()
