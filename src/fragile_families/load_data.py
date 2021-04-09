from typing import Tuple

import pandas as pd
from ballet.project import load_config
from funcy import some, where


def load_table_from_config(input_dir, table_config) -> pd.DataFrame:
    path = input_dir + '/' + table_config.path
    method = getattr(pd, table_config.pandas.read_method)
    kwargs = table_config.pandas.read_kwargs
    return method(path, **kwargs)


def load_data(
    input_dir=None, split='train',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data for a specific split

    If input_dir is not provided, loads X and y for the given split from the
    default location (S3). If input_dir is provided, loads the
    entities/targets tables from their default table names from the given
    directory, ignoring split.

    For feature development, only the train split should be used.
    """
    config = load_config()
    tables = config.data.tables
    entities_table_name = config.data.entities_table_name
    entities_config = some(where(tables, name=entities_table_name))
    targets_table_name = config.data.targets_table_name
    targets_config = some(where(tables, name=targets_table_name))

    if input_dir is None:
        bucket = config.data.s3_bucket
        split_path = config.data.get(split)
        input_dir = f's3://{bucket}/{split_path}'

    X = load_table_from_config(input_dir, entities_config)
    y = load_table_from_config(input_dir, targets_config)

    return X, y


def load_background() -> pd.DataFrame:
    """Load all background data as a single dataframe"""
    config = load_config()
    bucket = config.data.tables.s3_bucket
    path = f's3://{bucket}/raw/background.csv.gz'
    return pd.read_csv(path, compression='gzip', index_col='challengeID')
