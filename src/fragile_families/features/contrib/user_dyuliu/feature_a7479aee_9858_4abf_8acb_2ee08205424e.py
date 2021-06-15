# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "f4c26a",
    "f4c26b",
    "f4c26c",
    "f4c26d",
    "f4c26e",
    "m4c26a",
    "m4c26b",
    "m4c26c",
    "m4c26d",
    "m4c26e",
]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers


def transformer(df):
    ser1 = (
        df[["f4c26a", "f4c26b", "f4c26c", "f4c26d", "f4c26e"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser2 = (
        df[["m4c26a", "m4c26b", "m4c26c", "m4c26d", "m4c26e"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser3 = ser2 - ser1
    return ser3


# what is a brief name of this feature?
# type- str
name = "f4_buy_diff"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The difference between how mother and dad perceive the father's baby-relevant purchase behavior in year 5"

# put it all together!
feature = Feature(input, transformer, name, description)
