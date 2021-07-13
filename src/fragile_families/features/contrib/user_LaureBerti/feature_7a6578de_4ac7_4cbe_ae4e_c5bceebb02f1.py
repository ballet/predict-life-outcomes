# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import OneHotEncoder
import numpy as np
import ballet

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cm2fbir"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def check_age(row):
    if (row < 18).any():
        return 1
    elif (row > 20).any():
        return 2
    else:
        return 0


transformer = [
    ballet.eng.ValueReplacer(-9, 0),
    lambda df: df.apply(check_age, axis=1),
    OneHotEncoder(),
]

# what is a brief name of this feature?
# type- str
name = "check the age of the mother for her first birth <18, >21, and [18,21]"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "whether the mother is very (too) young for her first birth"
# put it all together!
feature = Feature(input, transformer, name=name, description=description)
