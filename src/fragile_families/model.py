from pathlib import Path
from ballet.eng.base import BaseTransformer

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

TARGETS = [
    'materialHardship',
    'gpa',
    'grit',
    'eviction',
    'layoff',
    'jobTraining',
]
DEFAULT_TARGET = 'materialHardship'

ENCODER_BLOCK_ID = 'fragile_families.encode_target#1'


def load_pipeline(
    name: str = DEFAULT_PIPELINE, target: str = DEFAULT_TARGET
) -> MLPipeline:
    if name not in PIPELINES:
        raise ValueError(f'Pipeline {name!r} is not supported')
    if target not in TARGETS:
        raise ValueError(f'Target {target!r} is not supported')
    init_params = {ENCODER_BLOCK_ID: {'target': target}}
    return MLPipeline(_load_pipeline(name), init_params=init_params)


def get_encoder_from_pipeline(pipeline: MLPipeline) -> BaseTransformer:
    """Get the target encoder instance from within the pipeline"""
    return pipeline.blocks[ENCODER_BLOCK_ID].instance
