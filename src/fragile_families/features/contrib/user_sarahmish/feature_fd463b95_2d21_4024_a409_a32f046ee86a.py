# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import ValueReplacer, NullFiller
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cm1hhinc", "cm2hhinc", "cm3hhinc", "cm4hhinc", "cm5hhinc", "m4f3a"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    ("cm1hhinc", ValueReplacer(-9, np.nan)),
    ("cm2hhinc", ValueReplacer(-9, np.nan)),
    ("cm3hhinc", ValueReplacer(-9, np.nan)),
    ("cm4hhinc", ValueReplacer(-9, np.nan)),
    ("cm5hhinc", ValueReplacer(-9, np.nan)),
    ("m4f3a", lambda x: np.where(x < 0, 1, x)),  # impute unanswered with 1
    # average income in all waves
    (
        ["cm1hhinc", "cm2hhinc", "cm3hhinc", "cm4hhinc", "cm5hhinc"],
        lambda df: df.mean(axis=1),
        "income",
    ),
    NullFiller(0),
    lambda x: x["income"] / x["m4f3a"],  # ratio
]

# what is a brief name of this feature?
# type- str
name = "HH income ratio"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "the ratio of household (HH) income by the number of people in HH surveyed in wave 4"

# put it all together!
feature = Feature(input, transformer, name, description)
