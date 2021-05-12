from ballet.eng.misc import ColumnSelector
from ballet.eng.missing import NullFiller
from ballet.transformer import make_robust_transformer_pipeline


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return make_robust_transformer_pipeline([
        ColumnSelector('materialHardship'),
        NullFiller(),
    ])
