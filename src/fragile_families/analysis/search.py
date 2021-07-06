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
from sklearn.base import clone
from stacklog import stacklog, stacktime

from fragile_families.analysis.evaluation import ROOT, SCORERS
from fragile_families.api import api
from fragile_families.model import (
    DEFAULT_TARGET, PIPELINES, TARGETS, get_encoder_from_pipeline,
    load_pipeline,)

SEP = '__'
DEFAULT_SEARCH_OUTPUT = ROOT.joinpath('data', 'output', 'search')
BTB_SESSION_MAX_ERRORS = 10
r2_holdout = SCORERS['r2_holdout']


class SessionManager:

    def __init__(self, session: BTBSession):
        self.session = session

    def save(self, output):
        sessionpath = output.joinpath('session.pkl')
        with stacklog(print, f'Dumping session to {sessionpath}'):
            with sessionpath.open('wb') as f:
                pickle.dump(self.session, f)

        resultspath = output.joinpath('results.json')
        with stacklog(print, f'Dumping results to {resultspath}'):
            with resultspath.open('w') as f:
                json.dump(self.session.proposals, f)

        bestpath = output.joinpath('best.json')
        with stacklog(print, f'Dumping best proposal to {bestpath}'):
            with bestpath.open('w') as f:
                json.dump(self.session.best_proposal, f)

    @classmethod
    def load(cls, name):
        sessionpath = DEFAULT_SEARCH_OUTPUT.joinpath(name, 'session.pkl')

        with stacklog(print, f'Loading session from {sessionpath}'):
            with sessionpath.open('rb') as f:
                return cls(pickle.load(f))

    @classmethod
    def latest(cls, target=DEFAULT_TARGET):
        latest_search = max(DEFAULT_SEARCH_OUTPUT.glob(f'search-{target}-*'))
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
    target: str,
    entities_tr, targets_tr, y_tr,
    entities_le, targets_le, y_le,
) -> Callable[[str, Dict], float]:

    def scorer(name, params):
        """Compute R2H of the pipeline with params on the leaderboard"""
        pipeline = pipelines[name]
        pipeline.set_hyperparameters(unflatten(params, sep=SEP))
        pipeline.fit(entities_tr, targets_tr)
        y_pred_le = pipeline.predict(entities_le)
        score = r2_holdout(y_le, y_pred_le, target)
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


def unflatten(doc: dict, sep: str = '.') -> dict:
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


def search(budget: int, output: Path, target: str):
    if output is None:
        output = DEFAULT_SEARCH_OUTPUT.joinpath(
            f'search-{target}-{datetime.now():%Y-%m-%d-%H-%M}')
    if output.exists() and output.is_file():
        raise ValueError('output must be a directory')
    else:
        output.mkdir(parents=True, exist_ok=True)

    pipelines = {
        name: load_pipeline(name, target=target)
        for name in PIPELINES
    }

    # get the encoder from within any one of the pipelines
    # the encoder should already have the correct target set
    encoder = clone(
        get_encoder_from_pipeline(
            pipelines[next(iter(pipelines.keys()))]))
    assert encoder.target == target

    with stacktime(print, 'Loading and encoding data'):
        entities_tr, targets_tr = api.load_data(split='train')
        entities_le, targets_le = api.load_data(split='leaderboard')
        y_tr = encoder.fit_transform(targets_tr)
        y_le = encoder.transform(targets_le)

    tunables = get_tunables(pipelines)
    scorer = make_scorer(
        pipelines,
        target,
        entities_tr, targets_tr, y_tr,
        entities_le, targets_le, y_le,
    )
    session = BTBSession(
        tunables, scorer,
        tuner_class=GCPTuner, max_errors=BTB_SESSION_MAX_ERRORS, verbose=True
    )

    with stacktime(print, f'Running session for {budget} iterations'):
        session.run(budget)

    session_manager = SessionManager(session)
    session_manager.save(output)


@click.command()
@click.option(
    '-b', '--budget',
    default=1,
    show_default=True,
    help='Number of iterations to search',
)
@click.option(
    '-o', '--output',
    default=None,  # will be set in main()
    help='Output directory for session results',
    type=Path,
)
@click.option(
    '-t', '--target',
    default=DEFAULT_TARGET,
    show_default=True,
    type=str,
)
@click.option(
    '-A', '--all-targets',
    is_flag=True,
    default=False,
    show_default=True,
)
def main(budget: int, output: Path, target: str, all_targets: bool):
    if all_targets:
        targets = TARGETS
    else:
        targets = [target]

    for _target in targets:
        search(budget, output, _target)


if __name__ == '__main__':
    main()
