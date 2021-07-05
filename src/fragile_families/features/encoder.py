from typing import Optional

from ballet.eng import BaseTransformer


class TargetSelector(BaseTransformer):
    """Target encoder for multi-target data to select one target at train time

    Can specify the desired target column either in the constructor or in the
    fit method kwargs. This is needed because within a pipeline, there is
    currently no ability to pass parameters to the constructor (the culprit is
    that the api.encoder object is obtained from calling get_target_encoder
    with no argument).

    At fit time, the target column is computed. At transform time, the target
    column is selected from the targets dataframe.
    """

    def __init__(self, default_target: Optional[str] = None):
        self.default_target = default_target

    def fit(self, y, target: Optional[str] = None):
        if target is not None:
            self.target_ = target
        elif self.default_target is not None:
            self.target_ = self.default_target
        else:
            raise ValueError('Need to specify either target or default_target')

        return self

    def transform(self, y):
        return y[self.target_]


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return TargetSelector(default_target='materialHardship')
