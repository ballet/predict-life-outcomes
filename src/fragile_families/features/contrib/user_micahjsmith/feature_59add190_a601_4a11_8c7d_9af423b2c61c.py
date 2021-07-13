# include any imports used in this feature right here (within this code cell)
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = None

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = None

# what is a brief name of this feature?
# type- str
name = None

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = None

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
