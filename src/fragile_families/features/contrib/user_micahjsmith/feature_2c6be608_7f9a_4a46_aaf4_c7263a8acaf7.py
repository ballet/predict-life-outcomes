# include any imports used in this feature right here (within this code cell)
from ballet import Feature
import numpy as np
import pandas as pd
from collections import Counter

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cm1ethrace", "cf1ethrace"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    # code all missing values to nan
    df = df.where(df > 0, 0)
    # map to vector of counts
    df = df.apply(lambda row: pd.Series(Counter(row.to_dict().values())), axis=1)
    # race/ethnicity categories as of ~2000
    cmap = {
        1: "white, non-hispanic",
        2: "black, non-hispanic",
        3: "hispanic",
        4: "other",
    }
    df = df.rename(columns=cmap)
    # remove 0 ('missing') category
    df = df.drop(0, axis=1, errors="ignore")
    # need to ensure all columns are there
    df = df.reindex(cmap.values(), axis=1)
    # fill missing with 0 counts
    df = df.fillna(0)
    return df


# what is a brief name of this feature?
# type- str
name = "race/ethnicity of child"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "encoding of race/ethnicity of child based on race/ethnicity of mother and father. each column represents race/ethnicity and the value represents the number of parents with that race/ethnicity."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
