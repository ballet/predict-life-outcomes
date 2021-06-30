# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullIndicator
from ballet.eng.external import SimpleImputer
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = "t5b1a"

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    # replace missing with nans
    lambda df: df.where(df > 0, np.nan),
    # impute missing
    SimpleImputer(strategy="mean"),
]

# what is a brief name of this feature?
# type- str
name = "child controls temper"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = None

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
