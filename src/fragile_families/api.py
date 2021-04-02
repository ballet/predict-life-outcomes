from ballet.project import FeatureEngineeringProject

import fragile_families as package
from fragile_families.features.encoder import get_target_encoder
from fragile_families.load_data import load_data


api = FeatureEngineeringProject(
    package=package,
    encoder=get_target_encoder(),
    load_data=load_data,
)
