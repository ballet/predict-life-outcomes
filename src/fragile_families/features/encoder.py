from ballet.eng.misc import ColumnSelector
from ballet.eng.missing import NullFiller


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return [
        ColumnSelector('materialHardship'),
        NullFiller(),
    ]
