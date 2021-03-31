from abc import ABC, abstractmethod
import numpy as np
from anomaly.changepoint import ChangePointTS
from tqdm.auto import tqdm

class Model(ABC):
    def __init__(self, transformer, feature_selector):
        self.transformer = transformer
        self.feature_selector = feature_selector
        self.predictions = None
        self.reals = None
        return

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def predict_next_day(self, X=None):
        if not self.transformer.predict_mode:
            raise SystemError('Model is not in eval mode')
        if X is None:
            X, Y_hist = self.prepare()
        preds = self._model.predict(X)
        return preds[-1]

    def predict_period(self, ts, cp_detection=False):
        if not self.transformer.predict_mode:
            raise SystemError('Model is not in eval mode')
        preds, reals = [], []
        cp = ChangePointTS()
        for i, row in tqdm(ts.iterrows(), total=ts.shape[0]):
            X, Y_hist = self.prepare()
            if cp_detection:
                idx = cp.detect(Y_hist, idx=Y_hist.index[-2],plot=False)
                if idx.size != 0:
                    raise SystemError('Change point detected!\n Dates: {}\n'.format(str(idx)))
            pred = self.predict_next_day(X)
            preds.append(pred)
            reals.append(row['Target'])
            self.fit_one_day(row)
        preds.append(self.predict_next_day())
        self.predictions = np.array(preds)
        self.reals = np.array(reals)
        return self.predictions
    
    def fit_one_day(self,row):
        self.transformer.step(row)
    
    def cross_validation(self, backward_window_size=200):
        pass

    def eval_bm(self):
        if self.reals is None:
            raise SystemError('Run model.predict_period() first')
        y_pred = self.predictions[:-1]
        y_test = self.reals
        value = 0
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                value += y_pred[i] * 0.005
                if y_pred[i] - y_test[i] >= 0:
                    value += (y_pred[i] - y_test[i]) * 0.009
                else:
                    value -= (y_pred[i] - y_test[i]) * 0.01
            else:
                value -= abs(y_pred[i] * 0.01)
                if y_test[i] - abs(y_test[i]) >= 0:
                    value += (y_test[i] - abs(y_pred[i])) * 0.009
                else:
                    value -= (y_test[i] - abs(y_pred[i])) * 0.01
        return value

    def mae(self):
        if self.reals is None:
            raise SystemError('Run model.predict_period() first')
        y_pred = self.predictions[:-1]
        y_test = self.reals

        return np.mean(np.abs(y_test - y_pred))

    def rmse(self):
        if self.reals is None:
            raise SystemError('Run model.predict_period() first')
        y_pred = self.predictions[:-1]
        y_test = self.reals

        return np.sqrt(np.mean((y_pred - y_test) ** 2))
    
    def eval(self):
        self.transformer.predict_mode = True