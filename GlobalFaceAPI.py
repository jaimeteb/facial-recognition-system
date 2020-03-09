"""
Servidor de API de reconocimiento facial.
"""

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.log import enable_pretty_logging
from tornado import autoreload, web, websocket

import cv2
import glob
import json
import momoko
import base64
import logging
import argparse
import numpy as np

from random import sample
from itertools import groupby

from lib.Augmentate import seq
from lib.FaceDetector import FaceDetectorDlib
from lib.FaceNetKeras import get_features, get_features_tolist
from lib.FaceAlign import get_aligned_face, get_landmarks_visualization
from lib.SSRNET_model import SSR_net, SSR_net_general
from lib.emotion_estimator import get_emotion_estimator_model


enable_pretty_logging()

autoreload.start()
[autoreload.watch(x) for x in glob.glob("./webapp/**/*.*", recursive=True)]

face_detector = FaceDetectorDlib()

"""
Configuración de redes neuronales para estimación de edad, género y emoción
"""

emo_dict = {0: "Enojado", 1: "Disgustado", 2: "Asustado", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorprendido"}
emo = get_emotion_estimator_model()

age_net = SSR_net(64, [3, 3, 3], 1, 1)()
age_net.load_weights('lib/weights/age_ssrnet_3_3_3_64_1.0_1.0.h5')

gender_net = SSR_net_general(64, [3, 3, 3], 1, 1)()
gender_net.load_weights('lib/weights/gender_ssrnet_3_3_3_64_1.0_1.0.h5')

def assert_face(crop):
    """
    Función que comprueba si hay un rostro en la imagen
    """
    return face_detector.detect(crop)

def get_emotion(image):
    """
    Función que estima la emoción de un rostro en una imagen
    """
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (48, 48))
    gray = np.expand_dims(np.expand_dims(gray, -1), 0)
    pred = emo.predict(gray)
    return emo_dict[np.argmax(pred)]

def get_age_gender(image):
    """
    Función que estima la edad y el género de un rostro en una imagen
    """
    agim = cv2.resize(image, (64, 64))
    agim = cv2.normalize(agim, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    agim = np.expand_dims(agim, axis=0)

    pred_age = age_net.predict(agim)[0]
    pred_gender = gender_net.predict(agim)[0]

    gender = 'Femenino' if pred_gender < 0.5 else 'Masculino'
    age = int(pred_age)

    return age, gender

class DataBaseHandler(web.RequestHandler):
    """
    Clase base para peticiones e interfaz con base de datos
    """

    def options(self):
        self.set_status(204)
        self.finish()

    @property
    def db(self):
        return self.application.db

    @property
    def clf(self):
        return self.application.clf

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")


class ListasHandler(DataBaseHandler):
    """
    Clase para manejar peticiones sobre las listas
    """

    @gen.coroutine
    def get(self):
        execstr = "select * from get_listas();"
        cursor = yield self.db.execute(execstr)
        listas = cursor.fetchall()
        self.write({
            "listas": listas
        })
        self.finish()


class TiendasHandler(DataBaseHandler):
    """
    Clase para manejar peticiones sobre las tiendas
    """

    @gen.coroutine
    def get(self):
        execstr = "select * from get_tiendas();"
        cursor = yield self.db.execute(execstr)
        tiendas = cursor.fetchall()
        self.write({
            "tiendas": tiendas
        })
        self.finish()


class FeaturesHandler(DataBaseHandler):
    """
    Clase para manejar peticiones sobre las características faciales
    """

    @gen.coroutine
    def get(self):
        """
        Función que obtiene todas las características faciales registradas
        """

        execstr = "select * from get_features();"
        cursor = yield self.db.execute(execstr)
        features = cursor.fetchall()
        self.write({
            "features": features
        })
        self.finish()

    @gen.coroutine
    def post(self):
        """
        Función que realiza la predicción de la identidad de un rostro
        """

        body = self.request.body
        postdata = json.loads(body)

        """
        Decodicicación de imagen
        """
        src = postdata["imagen"]
        img_decoded = base64.standard_b64decode(src)
        img_buffer = np.frombuffer(img_decoded, np.uint8)
        img = cv2.imdecode(img_buffer, 1)

        """
        Comprobación de rostro en la imagen
        """
        aface = assert_face(img)
        if aface:
            """
            Obtención de estimaciones del rostro
            """
            age, gender = get_age_gender(img)
            emotion = get_emotion(img)

            box = aface[0]
            crop = get_aligned_face(img, box)
            features = get_features_tolist(crop)
            idpred = self.clf.predict(features)

            if idpred is not None:
                """
                Predicción de identidad según ID encontrado
                """
                pred_id, sim = idpred
                execstr = "select name_from_id({});".format(pred_id)
                cursor = yield self.db.execute(execstr)
                pred = cursor.fetchall()[0][0]
            else:
                pred, sim = None, None

        else:
            age = gender = emotion = pred = sim = None

        self.write({
            'pred': pred,
            'sim': sim,
            'age': age,
            'gender': gender,
            'emotion': emotion
        })
        self.finish()

class RegistrosHandler(DataBaseHandler):
    """
    Clase para manejar peticiones sobre los registros
    """

    @gen.coroutine
    def put(self):
        """
        Función que maneja la inserción de nuevos registros
        """

        body = self.request.body
        putdata = json.loads(body)


        if putdata["action"] == "add":
            """
            En el caso que la acción sea 'add' se devuelve la visualización de
            los puntos faciales y el rostroo alineado, para comprobar que la
            imagen sea útil
            """

            imagenes = []
            for i, src in enumerate(putdata["imagenes"]):
                b64src = src[src.index(',') + 1:]
                b64src = b64src.encode('utf-8')
                imar = np.fromstring(base64.standard_b64decode(b64src), np.uint8)
                img = cv2.imdecode(imar, cv2.IMREAD_COLOR)
                imagenes.append(img)

            ret_imagenes = []
            for img in imagenes:
                box = face_detector.detect(img)
                if len(box) is 0:
                    ret_imagenes.append(("", ""))
                else:
                    box = box[0]
                    crop = get_aligned_face(img, box)
                    lands = get_landmarks_visualization(img, box)

                    _, enccrop = cv2.imencode('.jpg', crop)
                    b64crop = base64.b64encode(enccrop)
                    _, enclands = cv2.imencode('.jpg', lands)
                    b64lands = base64.b64encode(enclands)

                    ret_imagenes.append((
                        "data:image/jpeg;base64," + b64crop.decode("utf-8"),
                        "data:image/jpeg;base64," + b64lands.decode("utf-8")
                    ))

            ret = {
                "imagenes": ret_imagenes
            }

        elif putdata["action"] == "put":
            """
            En el caso que la acción sea 'put', se realiza aumento de imágenes
            hasta tener 10 imágenes por registro y se insertan en la base de
            datos
            """

            #TODO: CHECK FOR OTHER REGISTERS FOR SAME PERSON

            ids = []
            for name, regs in groupby(putdata["registros"], key=lambda x:x["nombre"]):
                ims = []
                augments = []
                ims_src = []
                for reg in list(regs):
                    src = reg["imagen"]
                    b64src = src[src.index(',') + 1:]
                    ims_src.append(b64src)
                    b64src = b64src.encode('utf-8')
                    imar = np.fromstring(base64.standard_b64decode(b64src), np.uint8)
                    img = cv2.imdecode(imar, cv2.IMREAD_COLOR)
                    ims.append(img)
                    augments.append("false")

                feats = [get_features_tolist(im) for im in ims]

                while len(ims) < 10:
                    img = sample(ims, 1)[0]

                    augim = seq.augment_images([img])[0]
                    box = face_detector.detect(augim)
                    if len(box) is not 0:
                        box = box[0]
                        crop = get_aligned_face(augim, box)
                        ims.append(crop)
                        feats.append(get_features_tolist(crop))

                        _, enccrop = cv2.imencode('.jpg', crop)
                        b64crop = base64.b64encode(enccrop)
                        ims_src.append(b64crop.decode("utf-8"))
                        augments.append("true")

                """
                Si no se proporciona un ID_GLOBAL se utiliza el siguiente en la
                tabla
                """
                if reg["id_global"] == "-1":
                    execstr = "select next_id_global();"
                    cursor = yield self.db.execute(execstr)
                    id_global = cursor.fetchall()[0][0]
                else:
                    id_global = reg["id_global"]

                execstr = "select * from put_registros({}, '{}', '{}', {}, {});".format(id_global, reg["nombre"], reg["comentarios"], reg["tienda"], reg["lista"])
                cursor = yield self.db.execute(execstr)
                id_registro = cursor.fetchall()[0][0]
                ids.append(id_registro)

                for src, fts, aug in zip(ims_src, feats, augments):
                    execstr = "select * from put_features({}, {}, '{}', array{}, {});".format(id_registro, id_global, src, fts, aug)
                    cursor = yield self.db.execute(execstr)
                    cursor.fetchall()

            ret = {
                "ids": ids
            }


            """
            Después de actualizar la base de datos, se actualiza el clasificador
            """
            execstr = "select * from get_features();"
            cursor = yield self.db.execute(execstr)
            features = cursor.fetchall()

            X = [fts for idg, fts in features]
            Y = [idg for idg, fts in features]

            self.clf.train(X, Y)
            self.clf.save()
            logging.info("Trained and saved classifier")


        self.write(ret)
        self.finish()

    @gen.coroutine
    def get(self):
        """
        Función que obtiene todos los registros
        """

        execstr = "select * from get_registros();"
        cursor = yield self.db.execute(execstr)
        registros = cursor.fetchall()

        registros_copy = registros.copy()
        for i, reg in enumerate(registros):
            execstr = "select * from get_mug({});".format(reg[0])
            cursor = yield self.db.execute(execstr)
            mug = cursor.fetchall()[0]
            registros_copy[i] = list(registros_copy[i])
            registros_copy[i].insert(2, mug)
            registros_copy[i] = tuple(registros_copy[i])

        self.write({
            "registros": registros_copy
        })
        self.finish()

    @gen.coroutine
    def delete(self):
        """
        Función que borra registros específicos
        """
        body = self.request.body
        deldata = json.loads(body)

        execstr = "select delete_registros({});".format(deldata["id_global"])
        cursor = yield self.db.execute(execstr)
        ok = cursor.fetchall()

        """
        Después de actualizar la base de datos, se actualiza el clasificador
        """
        execstr = "select * from get_features();"
        cursor = yield self.db.execute(execstr)
        features = cursor.fetchall()

        X = [fts for idg, fts in features]
        Y = [idg for idg, fts in features]

        self.clf.train(X, Y)
        self.clf.save()
        logging.info("Trained and saved classifier")

        self.write({
            "ok": ok
        })
        self.finish()


if __name__ == '__main__':
    """
    Se utiliza el clasificador especificado en la línea de comando
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier",
                        "-c",
                        help="Classifier (tree/svm)",
                        type=str,
                        default="svm")
    args = parser.parse_args()

    application = web.Application([
        (r'/api/registros', RegistrosHandler),
        (r'/api/listas', ListasHandler),
        (r'/api/tiendas', TiendasHandler),
        (r'/api/features', FeaturesHandler),
        (r'/(.*)', web.StaticFileHandler, {"path": "./webapp", "default_filename": "index.html"})
    ], debug=True)

    logging.info("\t\tApplication routes registered")

    ioloop = IOLoop.instance()

    """
    Conexión a la base de datos
    """
    application.db = momoko.Pool(
        dsn = "postgresql://postgres:postgres@postgres:5432/globalface",
        size = 1,
        ioloop = ioloop,
    )
    logging.info("\t\tDatabase defined")

    """
    Selección de clasificador
    """
    if args.classifier == "svm":
        from lib.SVMClassifier import Classifier
    elif args.classifier == "tree":
        from lib.TreeClassifier import Classifier

    application.clf = Classifier()
    application.clf.load()
    logging.info("\t\tClassifier loaded")

    future = application.db.connect()
    ioloop.add_future(future, lambda f: ioloop.stop())
    ioloop.start()
    future.result()
    logging.info("\t\tConnected to database")

    http_server = HTTPServer(application)
    http_server.listen(8000)
    logging.info("Serving at port 8000")
    ioloop.start()
