import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import click
import dill as pickle
from btb import BTBSession
from btb.tuning import GCPTuner, Tunable
from mlblocks import MLPipeline
from stacklog import stacklog, stacktime

from fragile_families.analysis.evaluation import ROOT, SCORERS
from fragile_families.api import api
from fragile_families.model import PIPELINES, load_pipeline

SEP = '__'
TARGET = 'materialHardship'
DEFAULT_SEARCH_OUTPUT = ROOT.joinpath('data', 'output', 'search')
BTB_SESSION_MAX_ERRORS = 10
r2_holdout = SCORERS['r2_holdout']


class SessionManager:

    def __init__(self, session: BTBSession, output: Path = None):
        self.session = session
        if output is None:
            self.output = DEFAULT_SEARCH_OUTPUT.joinpath(
                    f'search-{datetime.now():%Y-%m-%d-%H-%M}')
        else:
            self.output = output

    def save(self):
        sessionpath = self.output.joinpath('session.pkl')
        with stacklog(print, f'Dumping session to {sessionpath}'):
            with sessionpath.open('wb') as f:
                pickle.dump(self.session, f)

    @classmethod
    def load(cls, name):
        sessionpath = DEFAULT_SEARCH_OUTPUT.joinpath(name, 'session.pkl')

        with stacklog(print, f'Loading session from {sessionpath}'):
            with sessionpath.open('rb') as f:
                return cls(pickle.load(f))

    @classmethod
    def latest(cls):
        latest_search = max(DEFAULT_SEARCH_OUTPUT.glob('search-*'))
        return cls.load(latest_search.name)

    def best(self) -> MLPipeline:
        proposal = self.session.best_proposal
        pipeline = load_pipeline(proposal['name'])
        params = proposal['config']
        pipeline.set_hyperparameters(unflatten(params, sep=SEP))
        return pipeline


def get_tunables(pipelines: Dict[str, MLPipeline]):
    return {
        name: Tunable.from_dict(
            flatten(
                pipeline.get_tunable_hyperparameters(), sep=SEP, maxdepth=1))
        for name, pipeline in pipelines.items()
    }


def make_scorer(
    pipelines: Dict[str, MLPipeline],
    entities_tr, targets_tr, y_tr,
    entities_te, targets_te, y_te,
) -> Callable[[str, Dict], float]:

    def scorer(name, params):
        pipeline = pipelines[name]
        pipeline.set_hyperparameters(unflatten(params, sep=SEP))
        pipeline.fit(entities_tr, targets_tr)
        y_pred_te = pipeline.predict(entities_te)
        score = r2_holdout(y_te, y_pred_te, TARGET)
        return score

    return scorer


def flatten(doc: dict, sep: str = '.', maxdepth: int = sys.maxsize):
    """flatten keys in a dict"""

    def _join(prefix: str, k: str):
        return prefix + sep + k if prefix else k

    def _flatten(_doc: dict, prefix: str, depth: int):
        for k, v in _doc.items():
            if isinstance(v, dict) and depth < maxdepth:
                yield from _flatten(v, _join(prefix, k), depth+1)
            else:
                yield _join(prefix, k), v

    return dict(_flatten(doc, '', 0))


def unflatten(doc: dict, sep: str = '.'):
    """Recover dict from flattened keys

    In this simple implementation, require that each key is composed of just
    two parts.
    """
    # TODO: generalize this to arbitrarily nested keys
    result = {}
    for k, v in doc.items():
        prefix, rest = k.split(sep, maxsplit=1)
        assert sep not in rest
        if prefix not in result:
            result[prefix] = {}
        result[prefix][rest] = v
    return result


@click.command()
@click.option(
    '-b', '--budget',
    default=1,
    help='Number of iterations to search',
)
@click.option(
    '-o', '--output',
    default=lambda: DEFAULT_SEARCH_OUTPUT.joinpath(
        f'search-{datetime.now():%Y-%m-%d-%H-%M}'),
    help='Output directory for session results',
    type=Path,
)
def main(budget: int, output: Path):
    if output.exists() and output.is_file():
        raise ValueError('output must be a directory')
    else:
        output.mkdir(parents=True, exist_ok=True)

    pipelines = {
        name: load_pipeline(name)
        for name in PIPELINES
    }

    encoder = api.encoder
    with stacktime(print, 'Loading and encoding data'):
        entities_tr, targets_tr = api.load_data(split='train')
        entities_te, targets_te = api.load_data(split='leaderboard')
        y_tr = encoder.fit_transform(targets_tr)
        y_te = encoder.transform(targets_te)

    tunables = get_tunables(pipelines)
    scorer = make_scorer(
        pipelines,
        entities_tr, targets_tr, y_tr,
        entities_te, targets_te, y_te,
    )
    session = BTBSession(
        tunables, scorer,
        tuner_class=GCPTuner, max_errors=BTB_SESSION_MAX_ERRORS, verbose=True
    )

    with stacktime(print, f'Running session for {budget} iterations'):
        session.run(budget)

    sessionpath = output.joinpath('session.pkl')
    with stacklog(print, f'Dumping session to {sessionpath}'):
        with sessionpath.open('wb') as f:
            pickle.dump(session, f)

    resultspath = output.joinpath('results.json')
    with stacklog(print, f'Dumping results to {resultspath}'):
        with resultspath.open('w') as f:
            json.dump(session.proposals, f)

    bestpath = output.joinpath('best.json')
    with stacklog(print, f'Dumping best proposal to {bestpath}'):
        with bestpath.open('w') as f:
            json.dump(session.best_proposal, f)


if __name__ == '__main__':
    main()
