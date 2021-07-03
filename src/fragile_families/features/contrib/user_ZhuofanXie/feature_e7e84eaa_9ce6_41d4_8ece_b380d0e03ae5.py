# include any imports used in this feature right here (within this code cell)
from ballet import Feature
import numpy as np
import pandas as pd

# what are the input columns to this feature?
# type- Union[str, List[str]]
year2 = ["f2c18a", "f2c18b", "f2c18c", "f2c18d", "f2c18e", "f2c18f"]
year3 = ["f3c30a1", "f3c30a2", "f3c30a3", "f3c30a4", "f3c30a5"]
year4 = ["f4c26a", "f4c26b", "f4c26c", "f4c26d", "f4c26e"]
year5 = ["f5b22a", "f5b22b", "f5b22c", "f5b22d", "f5b22e", "f5b22f", "f5b22g", "f5b22h"]

input = year2 + year3 + year4 + year5

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    w2_avg = df[year2].clip(0, None).mean(axis=1)
    w3_avg = df[year3].clip(0, None).mean(axis=1)
    w4_avg = df[year4].clip(0, None).mean(axis=1)
    w5_avg = df[year5].clip(0, None).mean(axis=1)
    return np.array(
        pd.DataFrame({"w2": w2_avg, "w3": w3_avg, "w4": w4_avg, "w5": w5_avg}).std(
            axis=1
        )
    )


# what is a brief name of this feature?
# type- str
name = "father_buy_stab"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = (
    "The standard variation of father's baby-relevant purchases throughtout the waves."
)

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
