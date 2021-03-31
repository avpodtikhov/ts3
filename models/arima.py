from models.model import Model
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import pickle


class Arimer(Model):
    def __init__(self):
        super().__init__()
        self.__model = None
        self.__results = None

    def train(self, x_train, y_train):
        if len(x_train) != len(y_train): raise SystemError('Not Save Lengths')
        x_train.index = y_train.index
        self.__model = SARIMAX(y_train, exog=x_train, order=(1, 1, 1))
        self.__results = self.__model.fit()
        return self

    def predict(self, x_test, out_of_sample=False):
        y_pred = self.__results.forecast(len(x_test), exog=x_test, dynamic=True)
        return y_pred.values

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.__model, self.__results], f)

    def load(self, path):
        with open(path, "rb") as f:
            self.__model, self.__results = pickle.load(f)
