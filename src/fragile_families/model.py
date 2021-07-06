from pathlib import Path
from ballet.eng.base import BaseTransformer

from mlblocks import MLPipeline, add_pipelines_path, add_primitives_path
from mlblocks import load_pipeline as _load_pipeline

add_primitives_path(Path(__file__).parents[2].joinpath('blocks', 'primitives'))
add_pipelines_path(Path(__file__).parents[2].joinpath('blocks', 'pipelines'))


TRIVIAL_PIPELINES = [
    'train_mean',
    'leaderboard_mean',
    'test_mean',
]
REGRESSION_PIPELINES = [
    'ballet_randomforest',
    'ballet_xgboost',
    'ballet_elasticnet',
    'ballet_knn',
]
CLASSIFICATION_PIPELINES = [
    'ballet_randomforest_proba',
    'ballet_xgboost_proba',
    'ballet_elasticnet',
    'ballet_knn_proba',
]
PIPELINES = list(set([
    *TRIVIAL_PIPELINES,
    *REGRESSION_PIPELINES,
    *CLASSIFICATION_PIPELINES,
]))
DEFAULT_PIPELINE = 'ballet_randomforest'

TARGETS = [
    'materialHardship',
    'gpa',
    'grit',
    'eviction',
    'layoff',
    'jobTraining',
]
TARGETS_FANCY = {
    'materialHardship': 'Material Hardship',
    'gpa': 'GPA',
    'grit': 'Grit',
    'eviction': 'Eviction',
    'layoff': 'Layoff',
    'jobTraining': 'Job Training',
}
DEFAULT_TARGET = 'materialHardship'

TARGET_TYPES = {
    'materialHardship': 'regression',
    'gpa': 'regression',
    'grit': 'regression',
    'eviction': 'classification',
    'layoff': 'classification',
    'jobTraining': 'classification',
}

TARGET_PIPELINES = {
    'materialHardship': REGRESSION_PIPELINES,
    'gpa': REGRESSION_PIPELINES,
    'grit': REGRESSION_PIPELINES,
    'eviction': CLASSIFICATION_PIPELINES,
    'layoff': CLASSIFICATION_PIPELINES,
    'jobTraining': CLASSIFICATION_PIPELINES,
}

ENCODER_BLOCK_ID = 'fragile_families.encode_target#1'


def load_pipeline(
    name: str = DEFAULT_PIPELINE, target: str = DEFAULT_TARGET
) -> MLPipeline:
    if name not in PIPELINES:
        raise ValueError(f'Pipeline {name!r} is not supported')
    if target not in TARGETS:
        raise ValueError(f'Target {target!r} is not supported')
    if name not in (TRIVIAL_PIPELINES + TARGET_PIPELINES[target]):
        raise ValueError('Invalid pipeline for target')
    init_params = {ENCODER_BLOCK_ID: {'target': target}}
    return MLPipeline(_load_pipeline(name), init_params=init_params)


def get_encoder_from_pipeline(pipeline: MLPipeline) -> BaseTransformer:
    """Get the target encoder instance from within the pipeline"""
    return pipeline.blocks[ENCODER_BLOCK_ID].instance
