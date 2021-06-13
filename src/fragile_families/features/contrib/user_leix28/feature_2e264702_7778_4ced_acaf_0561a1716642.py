# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "t5b1a",
    "t5b1aa",
    "t5b1ab",
    "t5b1ac",
    "t5b1ad",
    "t5b1b",
    "t5b1c",
    "t5b1d",
    "t5b1e",
    "t5b1f",
    "t5b1g",
    "t5b1h",
    "t5b1i",
    "t5b1j",
    "t5b1k",
    "t5b1l",
    "t5b1m",
    "t5b1n",
    "t5b1o",
    "t5b1p",
    "t5b1q",
    "t5b1r",
    "t5b1s",
    "t5b1t",
    "t5b1u",
    "t5b1v",
    "t5b1w",
    "t5b1x",
    "t5b1y",
    "t5b1z",
]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    return df.values.clip(1, 4).mean(axis=1)


# what is a brief name of this feature?
# type- str
name = "t5_social"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "t5 social skills, average score of t5b1"

# put it all together!
feature = Feature(input, transformer, name, description)
