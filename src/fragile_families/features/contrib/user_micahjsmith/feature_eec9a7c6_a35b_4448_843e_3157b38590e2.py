# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import SimpleImputer, StandardScaler
import numpy as np

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = "t5c13c"

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df.where(df > 0, np.nan),
    # if there are all-nan columns, SimpleImputer will drop the rows
    lambda df: df if not df.isnull().all(axis=0).any() else df.fillna(3),
    SimpleImputer(strategy="mean"),
    StandardScaler(),
]

# what is a brief name of this feature?
# type- str
name = "child's year 9 math skills"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "teacher's assessment of child's math skills in year 9 compared to other students in the same grade"

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
