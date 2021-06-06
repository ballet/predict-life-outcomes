from ballet.eng import SimpleFunctionTransformer


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return SimpleFunctionTransformer(
        lambda df: df['materialHardship'].fillna(0)
    )
