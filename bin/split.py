#!/usr/bin/env python3

import sys
from collections import defaultdict

import click
import fsspec
import pandas as pd
from stacklog import stacktime


BUCKET = 'mit-dai-ballet-ff'
IDCOL = 'challengeID'
fs = fsspec.filesystem('s3')


def load(path):
    assert path.endswith('.csv.gz')
    with fs.open(f'{BUCKET}/{path}', 'rb') as f:
        return pd.read_csv(f, low_memory=False, compression='gzip')


def save(df, path):
    assert path.endswith('.parquet')
    with fs.open(f'{BUCKET}/{path}', 'wb') as f:
        df.to_parquet(f, index=True, engine='pyarrow')


def make_split(background, targets_raw):
    targets = targets_raw.set_index(IDCOL)
    entities = background.set_index(IDCOL).reindex(targets.index)
    return entities, targets


@stacktime(print, 'Creating splits')
def main():
    background = load('raw/background.csv.gz')
    train = load('raw/train.csv.gz')
    leaderboard = load('raw/leaderboard.csv.gz')
    test = load('raw/test.csv.gz')

    box = lambda: defaultdict(box)
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
            save(data[split][obj], f'data/{split}/{obj}.parquet')

if __name__ == '__main__':
    sys.exit(main())
