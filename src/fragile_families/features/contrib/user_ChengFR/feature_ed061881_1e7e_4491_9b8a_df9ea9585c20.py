# include any imports used in this feature right here (within this code cell)
import numpy as np
from ballet import Feature
from ballet.eng.external import SimpleImputer
from ballet.eng import ValueReplacer

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["f2k3m", "f3k2_13", "f4k2_13", "f5i2_13"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df.where(df >= 1, np.nan),
    ValueReplacer(2, 0),
    lambda df: df.max(axis=1),
    SimpleImputer(strategy="constant", fill_value=-1),
]

# what is a brief name of this feature?
# type- str
name = "Father's Job Training"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "Whether the father took any job skill program in wave 2-5."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
