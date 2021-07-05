from pathlib import Path

from mlblocks import MLPipeline, add_pipelines_path, add_primitives_path
from mlblocks import load_pipeline as _load_pipeline

add_primitives_path(Path(__file__).parents[2].joinpath('blocks', 'primitives'))
add_pipelines_path(Path(__file__).parents[2].joinpath('blocks', 'pipelines'))


PIPELINES = [
    'ballet_rf_regressor',
    'ballet_elasticnet',
    'train_mean',
    'leaderboard_mean',
    'test_mean',
]
DEFAULT_PIPELINE = 'ballet_rf_regressor'


def load_pipeline(name: str = DEFAULT_PIPELINE) -> MLPipeline:
    if name not in PIPELINES:
        raise ValueError(f'Pipeline {name!r} is not supported')
    return MLPipeline(_load_pipeline(name))
