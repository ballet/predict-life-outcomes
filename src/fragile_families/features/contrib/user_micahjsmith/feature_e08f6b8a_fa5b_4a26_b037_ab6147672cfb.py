# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import SimpleImputer, StandardScaler
import numpy as np
import pandas as pd

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["t5e1", "t5e3"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df.where(df > 0, np.nan),
    SimpleImputer(strategy="mean"),
    # in the edge case that the entire df is nans, simpleimputer drops all rows.
    # recover and lead to a ratio of 0/1 = 1.
    lambda arr: np.repeat([[0, 1]], arr.shape[0], axis=0) if arr.shape[1] == 0 else arr,
    # re-add column names :(
    lambda arr: pd.DataFrame(arr, columns=input),
    lambda df: df["t5e1"] / df["t5e3"],
    StandardScaler(),
]


# what is a brief name of this feature?
# type- str
name = "normalized student-teacher ratio"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "ratio of students to teachers (including adults like classroom aides) in the child's year 9 classroom, scaled to standard normal"

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
