"""Serializable feature transforms with training-fitted review reference dates."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReviewAgeTransformer(TransformerMixin, BaseEstimator):
    """Compute days since review against the latest training review date.

    Fitting the reference date prevents inference batch composition from
    changing predictions for the same record.
    """

    def fit(self, X, y=None):
        """Learn a fixed reference date from training rows."""
        self.reference_date_ = pd.to_datetime(np.asarray(X).ravel()).max()
        return self

    def transform(self, X):
        """Transform review dates without learning from validation/test rows."""
        dates = pd.to_datetime(np.asarray(X).ravel())
        return np.asarray((self.reference_date_ - dates).days).reshape(-1, 1)


def reshape_to_1d(values):
    """Flatten an imputed title column for TfidfVectorizer."""
    return np.asarray(values).ravel()
