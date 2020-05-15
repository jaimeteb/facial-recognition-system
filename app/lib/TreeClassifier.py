"""
Clasificador de Vecinos Cercanos con algoritmo de árbol
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

dot = lambda A, B: sum(a*b for a, b in zip(A, B))
cosine_similarity = lambda a, b: dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))

class Classifier():
    """
    Clase que define al clasificador de Vecinos Cercanos.
    """

    def __init__(self, threshold=0.8):
        self.clf = None
        self.threshold = threshold
        self.X = []
        self.Y = []

    def load(self):
        """
        Método para cargar un modelo de clasificador previamente guardado
        desde un archivo pickle.
        """
        if os.path.exists("lib/models/tree.pkl"):
            with open("lib/models/tree.pkl", "rb") as pkl:
                prev = pickle.load(pkl)
                self.clf = prev.clf
                self.X = prev.X
                self.Y = prev.Y
        else:
            self.clf = NearestNeighbors(algorithm='ball_tree', leaf_size=30)

    def save(self):
        with open("lib/models/tree.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    def train(self, X, Y):
        """
        Método para reentrenar el clasificador.
        """
        self.X, self.Y = X, Y
        self.clf = NearestNeighbors(algorithm='ball_tree', leaf_size=30)
        self.clf.fit(self.X)

    def predict(self, features):
        """
        Método para realizar una predicción de identidad según un vector de
        entrada.
        """
        vec = features
        try:
            dist, idx = self.clf.kneighbors([vec], 1)
        except sklearn.exceptions.NotFittedError:
            return None
        idx = idx.ravel()[0]
        pred = self.Y[idx]
        orig = self.X[idx]
        sim = cosine_similarity(vec, orig)

        if sim < self.threshold:
            return None
        else:
            return pred, sim
