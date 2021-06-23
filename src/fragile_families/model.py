from pathlib import Path

from mlblocks import MLPipeline, add_pipelines_path, add_primitives_path
from mlblocks import load_pipeline as _load_pipeline

add_primitives_path(Path(__file__).parents[2].joinpath('blocks', 'primitives'))
add_pipelines_path(Path(__file__).parents[2].joinpath('blocks', 'pipelines'))


def load_pipeline(name: str = 'ballet_rf_regressor') -> MLPipeline:
    pipeline_info = _load_pipeline(name)
    return MLPipeline(pipeline_info)
