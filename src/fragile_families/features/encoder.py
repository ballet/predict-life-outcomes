from ballet.eng import BaseTransformer


class TargetSelector(BaseTransformer):
    """Target encoder for multi-target data to select one target at predict time
    """

    def __init__(self, target: str):
        self.target = target

    def transform(self, y):
        return y[self.target] if y is not None else None


def get_target_encoder(target='materialHardship'):
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return TargetSelector(target)
