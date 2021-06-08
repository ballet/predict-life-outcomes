# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import ValueReplacer
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cm1age", "cf1age"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [("cf1age", ValueReplacer(-9, np.nan)), lambda x: x.max(axis=1)]

# what is a brief name of this feature?
# type- str
name = "age of parent"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "the age of the older parent when the survey took place"

# put it all together!
feature = Feature(input, transformer, name, description)
