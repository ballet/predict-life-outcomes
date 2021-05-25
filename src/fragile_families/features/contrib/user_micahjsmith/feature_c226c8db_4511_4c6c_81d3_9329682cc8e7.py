# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullFiller

# what are the input columns to this feature?
input = [
    "hv3d3",
    "hv3d10",
    "hv3d11",
    "hv3d12",
    "hv3d13",  # child hunger
]

# what transformations do you want to apply to these specific input columns?
transformer = [
    ("hv3d3", lambda ser: (ser == 1) | (ser == 2)),
    NullFiller(0),
    lambda df: df.sum(axis=1),
]

# what is a brief name of this feature?
name = "Children hungry wave 3"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
description = "Number of ways in which child may be measured as hungry in wave 3"

# put it all together!
feature = Feature(input, transformer, name, description)
