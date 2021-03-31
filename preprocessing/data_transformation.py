import pandas as pd
import numpy as np
from anomaly.detection import AnomalyDetector


class DataTransformer():
    def __init__(self, ts, lag_sz=5, detect_anomaly=True, predict_mode=False):
        self.lag_sz = lag_sz
        self.ts = ts
        self.predict_mode = predict_mode  # returns ONLY last no label lag
        
        self.calender = pd.read_csv('preprocessing/calendar.csv', index_col=0, parse_dates=True)

        self.detect_anomaly = detect_anomaly
        self.anomaly = AnomalyDetector(forward_window_size=5,
                                       threshold=3,
                                       drift=1.2)

        self.__Y_lags = self.__get_labels_lags()
        self.__X = self.__build_exog_features()
        self.__Y = self.__get_labels()

# public

    def getX(self):
        if self.predict_mode:
            return self.__X[-1:]
        return self.__X[:-1]

    def getY(self):
        return self.__Y

    def getY_lags(self):
        if self.predict_mode:
            return self.__Y_lags.iloc[-1:]
        return self.__Y_lags.iloc[:-1]
    
    def step(self, row):
        self.ts = self.ts.append(row)
        self.__Y_lags = self.__get_labels_lags()
        self.__X = self.__build_exog_features()
        self.__Y = self.__get_labels()

    @staticmethod
    def split_ts(timeseries, train_test_border):
        mask = None
        if type(train_test_border).__name__ in ['float', 'int']:
            mask = np.arange(
                len(timeseries)) < train_test_border * len(timeseries)
        else:
            mask = timeseries.index.values < train_test_border
        return timeseries[mask], timeseries[~mask]


# private

    def __build_exog_features(self):
        lag_mins = np.min(self.__Y_lags.values,
                          axis=1).reshape(-1, 1)  # среднее по строкам
        lag_means = np.mean(self.__Y_lags.values,
                            axis=1).reshape(-1, 1)  # среднее по строкам
        lag_maxs = np.max(self.__Y_lags.values,
                          axis=1).reshape(-1, 1)  # среднее по строкам
        lag_stds = np.std(self.__Y_lags.values,
                          axis=1).reshape(-1, 1)  # среднее по строкам
        lag_median = np.median(self.__Y_lags.values,
                          axis=1).reshape(-1, 1)  # среднее по строкам
        ex_features = np.concatenate((lag_mins, lag_means, lag_maxs, lag_stds, lag_median), axis=1)
        if self.detect_anomaly:
            for c in self.__Y_lags.columns:
                detected_anomalies = self.anomaly.detect(
                    self.__Y_lags[c],
                    return_series=False).reshape(-1, 1)  # детекция аномалий
                ex_features = np.concatenate((ex_features, detected_anomalies),
                                             axis=1)
        taxes = self.calender.loc[self.ts.index].values[self.lag_sz-1:]
        ex_features = np.concatenate((ex_features, taxes),
                                                     axis=1)
        # здесь можно добавить налоговый календарь - 0 если предсказываемый день обычный, 1 - если надо заплатить налог
        return ex_features

    def __get_labels(self):
        return pd.DataFrame(np.array(self.ts[self.lag_sz:]),
                            index=self.ts.index[self.lag_sz:])

    def __get_labels_lags(self):
        lags_list = [i.Target.values for i in self.ts.rolling(self.lag_sz)]
        return pd.DataFrame(
            np.array([i.Target.values for i in self.ts.rolling(self.lag_sz)
                      ][self.lag_sz - 1:]))