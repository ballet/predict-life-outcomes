# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "f3c30a1",
    "f3c30a2",
    "f3c30a3",
    "f3c30a4",
    "f3c30a5",
    "m3c30a",
    "m3c30b",
    "m3c30c",
    "m3c30d",
    "m3c30e",
]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers


def transformer(df):
    ser1 = (
        df[["f3c30a1", "f3c30a2", "f3c30a3", "f3c30a4", "f3c30a5"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser2 = (
        df[["m3c30a", "m3c30b", "m3c30c", "m3c30d", "m3c30e"]]
        .clip(0, None)
        .mean(axis=1)
    )
    ser3 = ser2 - ser1
    return ser3


# what is a brief name of this feature?
# type- str
name = "f3_buy_diff"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The difference between how mother and dad perceive the father's baby-relevant purchase behavior in year 3"

# put it all together!
feature = Feature(input, transformer, name, description)
