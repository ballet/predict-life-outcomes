# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import ValueReplacer
from ballet.util.log import logger
import numpy as np

# what are the input columns to this feature?
input = ["f2h17b", "f2h17c", "m2h19b", "m2h19c"]

# what transformations do you want to apply to these specific input columns?
transformer = [
    ValueReplacer(-9, np.nan),
    ValueReplacer(-6, np.nan),
    ValueReplacer(-2, np.nan),
    ValueReplacer(-1, np.nan),
    ValueReplacer(2, 0),  # no is 2
    (["f2h17b", "m2h19b"], lambda df: df.any(axis=1)),  # hungry children
    lambda df: df.sum(axis=1),  # hungry children or adults
]

# what is a brief name of this feature?
name = "Hungry Year 1"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
description = "Any indication someone in the family went hungry as of Year 1 survey"

# put it all together!
feature = Feature(input, transformer, name, description)
