# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import OneHotEncoder
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["t5c1"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    df["t5c1_yes"] = df[input] == 1
    df["t5c1_no"] = df[input] == 2
    df = df.drop(input, axis=1)
    df = df.astype(int)
    return df


# what is a brief name of this feature?
# type- str
name = "child is in special education classes"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = None

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
