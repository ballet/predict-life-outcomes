# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import OneHotEncoder
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["t5d1a", "t5d1b"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def conference_or_meeting(row):
    if (row == 1).any():
        return 1
    elif (row == 2).any():
        return 2
    else:
        return 0


transformer = [
    lambda df: df.apply(conference_or_meeting, axis=1),
    OneHotEncoder(),
]

# what is a brief name of this feature?
# type- str
name = "parents attended conferences or meetings with teacher"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "whether parents attended conferences or meetings with teacher, whether they did *not* attend, or other (i.e. unknown)"

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
