# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "f2b17a",
    "f2b17b",
    "f2b17c",
    "f2b17d",
    "f2b17e",
    "f2b17f",
    "f2b17g",
    "f2b17h",
    "f2b36a",
    "f2b36b",
    "f2b36c",
    "f2b36d",
    "f2b36e",
    "f2b36f",
    "f2b36g",
    "f2b36h",
]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = None


def transformer(df):
    return df.mean(axis=1)


# what is a brief name of this feature?
# type- str
name = "f1iwc_nobound"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "father's interaction with children in the first year"

# put it all together!
feature = Feature(input, transformer, name, description)
