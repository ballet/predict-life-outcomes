# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "f2c18a",
    "f2c18b",
    "f2c18c",
    "f2c18d",
    "f2c18e",
    "f2c18f",
    "m2c23a",
    "m2c23b",
    "m2c23c",
    "m2c23d",
    "m2c23e",
    "m2c23f",
]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers


def transformer(df):
    ser1 = (
        df[["f2c18a", "f2c18b", "f2c18c", "f2c18d", "f2c18e", "f2c18f"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser2 = (
        df[["m2c23a", "m2c23b", "m2c23c", "m2c23d", "m2c23e", "m2c23f"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser3 = ser2 - ser1
    return ser3


# what is a brief name of this feature?
# type- str
name = "f2_buy_diff"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The difference between how mother and dad perceive the father's baby-relevant purchase behavior in year 1"

# put it all together!
feature = Feature(input, transformer, name, description)
