#!/usr/bin/env python3

import sys
from collections import defaultdict

import fsspec
import numpy as np
import pandas as pd
from stacklog import stacktime


IDCOL = 'challengeID'


def load(fs, root, path):
    assert path.endswith('.csv.gz')
    with fs.open(f'{root}/{path}', 'rb') as f:
        df = pd.read_csv(f, low_memory=False, compression='gzip')

    # clean `TRUE` and `FALSE` labels in raw targets frames
    for col in ['layoff', 'jobTraining', 'eviction']:
        if col in df and not np.issubdtype(df[col].dtype, np.number):
            oldsum = np.sum(df[col])
            df[col] = df[col].map({True: 1, False: 0, None: np.nan})
            assert np.sum(df[col]) == oldsum

    return df


def save(fs, root, df, path):
    assert path.endswith('.parquet')
    with fs.open(f'{root}/{path}', 'wb') as f:
        df.to_parquet(f, index=True, engine='pyarrow')


def make_split(background, targets_raw):
    targets = targets_raw.set_index(IDCOL)
    entities = background.set_index(IDCOL).reindex(targets.index)
    return entities, targets


def box():
    return defaultdict(box)


@stacktime(print, 'Creating splits')
def main(fs, root):
    background = load(fs, root, 'raw/background.csv.gz')
    train = load(fs, root, 'raw/train.csv.gz')
    leaderboard = load(fs, root, 'raw/leaderboard.csv.gz')
    test = load(fs, root, 'raw/test.csv.gz')

    # the leaderboard labels also include rows for other challenge datasets
    leaderboard = leaderboard[
        ~(
            leaderboard['challengeID'].isin(train['challengeID'])
            | leaderboard['challengeID'].isin(test['challengeID'])
        )
    ]

    data = box()
    for split, labels in (
        ('train', train),
        ('leaderboard', leaderboard),
        ('test', test),
    ):

        data[split]['entities'], data[split]['targets'] = \
            make_split(background, labels)

    for split in ('train', 'leaderboard', 'test'):
        for obj in ('entities', 'targets'):
            save(fs, root, data[split][obj], f'data/{split}/{obj}.parquet')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _fs, root = sys.argv[1].split('://')
        fs = fsspec.filesystem(_fs)
    else:
        fs = fsspec.filesystem('s3')
        root = 'mit-dai-ballet-ff'
    sys.exit(main(fs, root))
