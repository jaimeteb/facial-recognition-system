"""
Clasificador de tipo Support Vector Machine.
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC

class Classifier():
    """
    Clase que define el clasificador SVM
    """

    def __init__(self, threshold=0.3):
        self.clf = None
        self.threshold = threshold
        self.X = []
        self.Y = []

    def load(self):
        """
        Método para cargar un modelo de clasificador previamente guardado
        desde un archivo pickle.
        """
        if os.path.exists("lib/models/svm.pkl"):
            with open("lib/models/svm.pkl", "rb") as pkl:
                prev = pickle.load(pkl)
                self.clf = prev.clf
                self.X = prev.X
                self.Y = prev.Y
        else:
            self.clf = SVC(C = 1.0,
                           kernel = 'linear',
                           probability = True)

    def save(self):
        with open("lib/models/svm.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    def train(self, X, Y):
        """
        Método para reentrenar el clasificador.
        Se entrena hasta que el número de clases es mayor a 1.
        """
        self.X, self.Y = X, Y
        self.clf = SVC(C = 1.0,
                       kernel = 'linear',
                       probability = True)
        if len(self.Y) > 1:
            self.clf.fit(self.X, self.Y)

    def predict(self, vec):
        """
        Método para realizar una predicción de identidad según un vector de
        entrada.
        """
        try:
            id = self.clf.predict([vec])[0]
        except sklearn.exceptions.NotFittedError:
            return None

        probs = self.clf.predict_proba([vec])[0].tolist()
        prob = probs[self.clf.classes_.tolist().index(id)]

        if prob < self.threshold:
            return None
        else:
            return id, prob
