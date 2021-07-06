from ballet import Feature
import numpy as np
import pandas as pd

# what are the input columns to this feature?
# type- Union[str, List[str]]
year2 = ["m2c23a", "m2c23b", "m2c23c", "m2c23d", "m2c23e", "m2c23f"]
year3 = ["m3c30a", "m3c30b", "m3c30c", "m3c30d", "m3c30e"]
year4 = ["m4c26a", "m4c26b", "m4c26c", "m4c26d", "m4c26e"]
year5 = [
    "m5b22a",
    "m5b22b",
    "m5b22c",
    "m5b22d",
    "m5b22e",
    "m5b22f",
    "m5b22g",
    "m5b22h",
    "m5b22i",
]

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
name = "father_buy_stab_in_mothers_view"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The standard deviation of father's baby-relevant purchases in mother's view throughtout the waves."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
