# include any imports used in this feature right here (within this code cell)
from ballet.eng.external import SimpleImputer
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "cf1finjail",
    "cf2fevjail",
    "cf2finjail",
    "cf3fevjail",
    "cf3finjail",
    "cf4fevjail",
    "cf4finjail",
    "cf5fevjail",
    "cf5finjail",
]


# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df[
        [
            "cf1finjail",
            "cf2fevjail",
            "cf2finjail",
            "cf3fevjail",
            "cf3finjail",
            "cf4fevjail",
            "cf4finjail",
            "cf5fevjail",
            "cf5finjail",
        ]
    ].max(axis=1),
    SimpleImputer(strategy="most_frequent"),
]

# what is a brief name of this feature?
# type- str
name = "Father incarcerated"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "Father ever incarcerated."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
