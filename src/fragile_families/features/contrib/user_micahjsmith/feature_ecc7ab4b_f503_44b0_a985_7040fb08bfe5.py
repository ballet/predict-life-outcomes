# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["m4i23b", "m4i23c", "f4i23b", "f4i23c"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    ("m4i23c", lambda ser: ser == 1),  # yes labeled as 1
    ("f4i23c", lambda ser: ser == 1),  # yes labeled as 1
    ("m4i23b", lambda ser: ser == 1),  # yes labeled as 1
    ("f4i23b", lambda ser: ser == 1),  # yes labeled as 1
    (
        ["m4i23c", "f4i23c"],
        lambda df: df.any(axis=1),
    ),  # either father/mother can report child(ren) hunger
    lambda df: df.sum(axis=1),
]

# what is a brief name of this feature?
# type- str
name = "Hungry wave 4"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = (
    "Degree to which father, mother, or child(ren) experienced hunger in wave 4"
)

# put it all together!
feature = Feature(input, transformer, name, description)
