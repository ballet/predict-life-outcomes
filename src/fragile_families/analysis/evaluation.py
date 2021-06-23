import logging
from pathlib import Path
from typing import Tuple

from ballet import b
from mlblocks import MLPipeline
from sklearn.metrics import mean_squared_error
from timer_cm import Timer as _Timer

try:
    import seaborn as sns
except ImportError:
    sns = None

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3].joinpath('evaluation')
SPLITS = ('train', 'leaderboard', 'test')
TRAIN_MSE = {
    'materialHardship': 0.025,
    'gpa': 0.425,
    'grit': 0.253,
    'eviction': 0.056,
    'layoff': 0.167,
    'jobTraining': 0.185,
}
TARGETS = list(TRAIN_MSE.keys())


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


def evaluate(pipeline: MLPipeline, split='leaderboard'):
    encoder = b.api.encoder
    data = {split: {} for split in SPLITS}
    for split in SPLITS:
        data[split]['X_df'], data[split]['y_df'] = b.api.load_data(split=split)
    return _evaluate(pipeline, encoder, data, SCORERS)


def _evaluate(pipeline, encoder, data, scorers) -> Tuple[MLPipeline, dict]:
    with Timer('evaluate', print_results=False) as timer:
        with timer.child('fit'):
            pipeline.fit(data['train']['X_df'], data['train']['y_df'])
            encoder.fit(data['train']['y_df'])

        scores = {split: {} for split in SPLITS}

        for split in SPLITS:
            split_timer = timer.child(split)
            with split_timer:
                with split_timer.child('predict'):
                    y_pred = pipeline.predict(data[split]['X_df'])
                y = encoder.transform(data[split]['y_df'])
                with split_timer.child('score'):
                    for scorer in scorers:
                        name = scorer.__name__ or 'unknown'
                        if name != 'r2_holdout':
                            scores[split][name] = scorer(y, y_pred)
                        else:
                            scores[split][name] = \
                                scorer(y, y_pred, 'materialHardship')

    results = {
        'scores': scores,
        'timing': timer.details(),
    }

    return pipeline, results


def r2_holdout(y, y_pred, target):
    return 1.0 - mean_squared_error(y, y_pred) / TRAIN_MSE[target]


SCORERS = [
    mean_squared_error,
    r2_holdout,
]
