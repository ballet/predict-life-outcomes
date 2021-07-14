# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullFiller, ValueReplacer
import ballet

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = ["cm1relf", "cm2relf"]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    ballet.eng.ValueReplacer(-9, 0),
    ValueReplacer(1, 0),
    ValueReplacer(2, 0),
    ValueReplacer(3, 0),
    ValueReplacer(4, 0),
    ValueReplacer(5, 1),
    ValueReplacer(6, 0),
    ValueReplacer(7, 1),
    ValueReplacer(8, 1),
    NullFiller(0),
]

# what is a brief name of this feature?
# type- str
name = "Parental romantic relation"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "Lack of romantic relation between parents (separation/divorce/death/no relation/dad unknown)"

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
