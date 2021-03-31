from models.model import Model
from catboost import CatBoostRegressor, Pool
import numpy as np

class CatBooster(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    def train(self, train_ts):
        x_train, y_train = self.prepare()

        train_p = Pool(x_train, y_train)

        model = CatBoostRegressor(verbose=False)
        self._model = model.fit(train_p)

        return self
    
    def prepare(self):
        X = self.transformer.getX()
        Y = self.transformer.getY()
        Y_lags = self.transformer.getY_lags()
        if self.feature_selector:
            if self.transformer.predict_mode:
                X = self.feature_selector.predict(X)
            else:
                X = self.feature_selector.fit_predict(X, Y)
                print('only features columns', self.feature_selector.get_mask(), 'was selected from X_train')
            
        X = np.concatenate((Y_lags, X), axis=1)
        return X, Y

    def save(self, path):
        self._model.save_model(path)

    def load(self, path):
        model = CatBoostRegressor()
        self._model = model.load_model(path)
