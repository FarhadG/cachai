import numpy as np
from river import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScaler():
    def __init__(self):
        self._scaler = preprocessing.StandardScaler()

    def _iterate(self, X):
        for i in X[0]:
            yield {f'feature{i}': i}

    def partial_fit(self, X):
        for feature in self._iterate(X):
            self._scaler.learn_one(feature)

    def transform(self, X):
        output = []
        for feature in self._iterate(X):
            X_scaled = self._scaler.transform_one(feature)
            X_scaled = list(X_scaled.values())
            output.append(X_scaled)
        return output

    def fit_transform(self, X):
        output = []
        for feature in self._iterate(X):
            self._scaler.learn_one(feature)
            X_scaled = self._scaler.transform_one(feature)
            X_scaled = list(X_scaled.values())
            output.append(X_scaled)
        return output


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

    def partial_fit(self, X):
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.var_ = np.var(X, axis=0)
            self.n_samples_seen_ = X.shape[0]
        else:
            n_samples = X.shape[0]
            new_mean = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)

            total_samples = self.n_samples_seen_ + n_samples
            updated_mean = (self.n_samples_seen_ * self.mean_ + n_samples * new_mean) / total_samples

            self.var_ = (self.n_samples_seen_ * (self.var_ + (self.mean_ - updated_mean) ** 2) +
                         n_samples * (new_var + (new_mean - updated_mean) ** 2)) / total_samples
            self.mean_ = updated_mean
            self.n_samples_seen_ = total_samples
        return self

    def transform(self, X):
        if self.mean_ is None or self.var_ is None:
            raise ValueError('Scaler has not been fitted yet.')
        epsilon = 1e-9  # Avoid division by zero
        return (X - self.mean_) / (np.sqrt(self.var_) + epsilon)

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)
