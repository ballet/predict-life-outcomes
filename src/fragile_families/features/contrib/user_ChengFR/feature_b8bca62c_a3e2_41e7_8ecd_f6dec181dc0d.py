# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import SimpleImputer, MinMaxScaler

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cf1edu", "cm1edu"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df[["cf1edu", "cm1edu"]].max(axis=1),
    lambda df: df >= 4,
    SimpleImputer(strategy="mean"),
]

# what is a brief name of this feature?
# type- str
name = "college-educated parents"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "Whether the father or the monther has a degree of college education or above in wave 1."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
