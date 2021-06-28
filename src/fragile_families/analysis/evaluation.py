import logging
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Tuple

from ballet import b
from mlblocks import MLPipeline
import numpy as np
from sklearn.metrics import mean_squared_error
from timer_cm import Timer as _Timer

try:
    import seaborn as sns
except ImportError:
    sns = None

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3].joinpath('evaluation')
SPLITS = ('train', 'leaderboard', 'test')
TRAIN_MEAN = {
    'gpa': 2.866738,
    'grit': 3.427539,
    'materialHardship': 0.10374478160633066,
    'eviction': 0.059630,
    'layoff': 0.209084,
    'jobTraining': 0.234771,
}
TARGETS = list(TRAIN_MEAN.keys())


def box():
    return defaultdict(box)


def unbox(box):
    result = {}
    for k, v in box.items():
        result[k] = unbox(v) if isinstance(v, defaultdict) else v
    return result


def set_style():
    if sns is not None:
        sns.set_theme(
            context='paper', style='whitegrid', font='serif', font_scale=1.5)
    else:
        logger.warning('Seaborn is not installed')


def savefig(fig, name: str, figdir: str):
    figdir = Path(figdir)
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(
            str(figdir / (name + ext)), bbox_inches='tight', pad_inches=0)


def savetable(
    df, name: str, tabledir: str, csv_kwargs=None, latex_kwargs=None
):
    if csv_kwargs is None:
        csv_kwargs = {}
    if latex_kwargs is None:
        latex_kwargs = {}
    latex_kwargs.setdefault('float_format', '{:0.3f}'.format)
    tabledir = Path(tabledir)
    df.to_csv(tabledir / f'{name}.csv', **csv_kwargs)
    df.to_latex(tabledir / f'{name}_tabular.tex', **latex_kwargs)


def camelcase(s: str, sep='_'):
    parts = s.split(sep)
    parts = [part.title() for part in parts]
    if parts and parts[0]:
        parts[0] = parts[0][0].lower() + parts[0][1:]
    return ''.join(parts)


class Timer(_Timer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('print_results', False)
        super().__init__(*args, **kwargs)

    def child(self, name):
        try:
            return self._children[name]
        except KeyError:
            result = Timer(name, print_results=False)
            self._children[name] = result
            return result

    def details(self):
        children = self._children.values()
        elapsed = self.elapsed or sum(c.elapsed for c in children)
        if not self._children:
            return {self._name: float(elapsed)}
        result = {self._name: {'total': float(elapsed),}}
        for child in children:
            result[self._name].update(child.details())
        return result


def evaluate(pipeline: MLPipeline, splits=None):
    if splits is None:
        splits = SPLITS
    encoder = b.api.encoder
    data = defaultdict(dict)
    for split in splits:
        data[split]['X_df'], data[split]['y_df'] = b.api.load_data(split=split)
    data = unbox(data)
    return _evaluate(pipeline, encoder, data, SCORERS)


def _evaluate(pipeline, encoder, data, scorers) -> Tuple[MLPipeline, dict]:
    with Timer('evaluate', print_results=False) as timer:
        with timer.child('fit'):
            pipeline.fit(data['train']['X_df'], data['train']['y_df'])
            encoder.fit(data['train']['y_df'])

        scores = box()

        for split in SPLITS:
            split_timer = timer.child(split)
            with split_timer:
                with split_timer.child('predict'):
                    y_pred = pipeline.predict(data[split]['X_df'])
                y = encoder.transform(data[split]['y_df'])
                with split_timer.child('score'):
                    for scorer_name, scorer in scorers.items():
                        if scorer_name != 'r2_holdout':
                            scores[split][scorer_name] = scorer(y, y_pred)
                        else:
                            scores[split][scorer_name] = \
                                scorer(y, y_pred, 'materialHardship')

    results = {
        'scores': unbox(scores),
        'timing': timer.details(),
    }

    return pipeline, results


def r2_holdout(y, y_pred, target):
    return 1.0 - mean_squared_error(y, y_pred) / mean_squared_error(
        y, np.full_like(y, TRAIN_MEAN[target]))


def skipna(func):

    @wraps(func)
    def wrapped(y, y_pred, *args, **kwargs):
        inds = ~np.isnan(y)
        y = y[inds]
        y_pred = y_pred[inds]
        return func(y, y_pred, *args, **kwargs)

    return wrapped


SCORERS = {
    'mean_squared_error': skipna(mean_squared_error),
    'r2_holdout': skipna(r2_holdout),
}
SCORER_FANCY_NAMES = {
    'mean_squared_error': 'MSE',
    'r2_holdout': r'$R^2_{\text{Holdout}}$',
}


def move_column(df, name, index=0, before=None, after=None):
    if name not in df.columns:
        raise ValueError(f'Column {name!r} not in DataFrame')

    df = df.copy()
    col = df[name]
    del df[name]

    if before is not None:
        index = list(df.columns).index(before)
    elif after is not None:
        index = list(df.columns).index(after) + 1

    df.insert(index, name, col)
    return df
