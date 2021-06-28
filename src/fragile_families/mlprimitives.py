import numpy as np
from ballet.eng import BaseTransformer


class DropMissingTargets(BaseTransformer):

    def fit(self, X, y, **fit_kwargs):
        self.inds_ = ~np.isnan(y)

    def transform(self, X, y):
        if y is not None:
            if hasattr(X, 'loc'):
                return X.loc(axis=0)[self.inds_], y[self.inds_]
            else:
                return X[self.inds_, :], y[self.inds_]
        else:
            return X, y
