from ballet.eng import ColumnSelector


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return ColumnSelector('materialHardship')
