# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "f5b22a",
    "f5b22b",
    "f5b22c",
    "f5b22d",
    "f5b22e",
    "f5b22f",
    "f5b22g",
    "f5b22h",
    "f5b22i",
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

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers


def transformer(df):
    ser1 = (
        df[
            [
                "f5b22a",
                "f5b22b",
                "f5b22c",
                "f5b22d",
                "f5b22e",
                "f5b22f",
                "f5b22g",
                "f5b22h",
                "f5b22i",
            ]
        ]
        .clip(0, None)
        .mean(axis=1)
    )
    ser2 = (
        df[
            [
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
        ]
        .clip(0, None)
        .mean(axis=1)
    )
    ser3 = ser2 - ser1
    return ser3


# what is a brief name of this feature?
# type- str
name = "f5_buy_diff"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The difference between how mother and dad perceive the father's baby-relevant purchase behavior in year 9"

# put it all together!
feature = Feature(input, transformer, name, description)
