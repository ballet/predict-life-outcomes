# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import ValueReplacer
from ballet.util.log import logger
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["f2h17b", "f2h17c", "m2h19b", "m2h19c"]


def consolidate_hungry_children(df):
    cols = ["f2h17b", "m2h19b"]
    df["hungry_children"] = df.loc[:, cols].any(axis=1)
    for col in cols:
        del df[col]
    return df


# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform methods), or list of functions/transformers
transformer = [
    ValueReplacer(-9, np.nan),
    ValueReplacer(-6, np.nan),
    ValueReplacer(-2, np.nan),
    ValueReplacer(-1, np.nan),
    ValueReplacer(2, 0),  # no is 2
    consolidate_hungry_children,
    lambda df: df.sum(axis=1),
]

# what is a brief name of this feature?
# type- str
name = "Hungry Year 1"

# what is a longer human-readable description for this feature? you can include more background on your calculations or thinking
# type- Optional[str]
description = "Any indication someone went hungry as of Year 1 survey"

# put it all together!
feature = Feature(input, transformer, name, description)
