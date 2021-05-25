# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullFiller
from ballet.eng.external import StandardScaler


def often_or_sometimes_true(ser):
    return (ser == 1) | (ser == 2)


# what are the input columns to this feature?
input = [
    "hv3d1a",
    "hv3d1b",
    "hv3d1c",
    "hv3d4",
    "hv3d5",
    "hv3d6",
    "hv3d7",
    "hv3d9",
]

# what transformations do you want to apply to these specific input columns?
transformer = [
    ("hv3d1a", often_or_sometimes_true),
    ("hv3d1b", often_or_sometimes_true),
    ("hv3d1c", often_or_sometimes_true),
    lambda df: df.sum(axis=1),
]

# what is a brief name of this feature?
name = "Adults hungry wave 3"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
description = "Number of ways in which adults may be measured as hungry in wave 3"

# put it all together!
feature = Feature(input, transformer, name, description)
