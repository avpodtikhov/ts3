import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

import pickle


class FeatureSelector:
    def __init__(self, model, algorithm_name: str = '1', lag_sz=5, pretrained_path=None):
        self.model = model()

        self.method = self.__choose_method(algorithm_name)
        self.lag_sz = lag_sz
        self.all_features = None
        self.selected_features = None
        if pretrained_path is not None:
            with open(pretrained_path, 'rb') as f:
                self.all_features, self.selected_features = pickle.load(f)

    def fit_predict(self, X, y):
        self.all_features = np.arange(X.shape[1])
        self.selected_features = self.method(X, y)
        return self.predict(X)

    def predict(self, X):
        if X.shape[1] != len(self.all_features):
            raise SystemError('Selecting features not from the same features-list as was learned')
        selection_mask = self.get_mask()
        return pd.DataFrame(X)[selection_mask]

    def get_mask(self):
        if self.selected_features is None:
            raise SystemError('First fit it please by use fit_predict_method')
        return self.selected_features

    def save(self, path):
        if path is None:
            return
        with open(path, 'wb') as f:
            pickle.dump([self.all_features, self.selected_features], f)

    def __filter_target_features(self, X, y, corr_coef=0.5):
        df = pd.DataFrame(X, columns=self.all_features)
        df['y'] = y.values
        cor = df.corr()
        cors = abs(cor['y'])
        cors_without_target = cors.filter(items=self.all_features)
        relevant_features = cors_without_target[cors_without_target > corr_coef].keys().values
        return relevant_features

    def __backward_elimination(self, X, y, significance_level=0.05):
        features = self.all_features.copy()
        while len(features) > 0:
            features_with_constant = sm.add_constant(X[:, features])
            p_values = sm.OLS(y, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if max_p_value >= significance_level:
                excluded_feature = np.argmax(p_values.values)
                features = np.delete(features, excluded_feature)
            else:
                break
        return features

    def __forward_selection(self, X, y, significance_level=0.05):
        initial_features = self.all_features.copy()
        best_features = []
        while len(initial_features) > 0:
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[:, best_features + [new_column]])).fit()
                new_pval.loc[new_column] = model.pvalues.iloc[-1]
            min_p_value = new_pval.min()
            if min_p_value < significance_level:
                best_features.append(new_pval.index[np.argmin(new_pval.values)])
            else:
                break
        return sorted(best_features)

    # no of features
    def __RFE(self, X, y):
        high_score = 0
        best_features = None
        score_list = []
        for n in np.arange(1, 15):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            rfe = RFE(self.model, n)
            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_test_rfe = rfe.transform(X_test)
            self.model.train(X_train_rfe, y_train)
            score = self.model.eval_mae(X_test_rfe, y_test)
            score_list.append(score)
            if score > high_score:
                high_score = score
                best_features = X_train.columns[rfe.support_]
                print(f"Founded more optimum number of features: {n}")
            print("Score with %d features: %f" % (n, score))
        return best_features

    def __choose_method(self, algo_name):
        if algo_name == '1':
            return self.__filter_target_features
        elif algo_name == '2':
            return self.__backward_elimination
        elif algo_name == '3':
            return self.__forward_selection
        elif algo_name == '4':
            return self.__RFE
        else:
            raise InterruptedError('Wrong algorith name')
