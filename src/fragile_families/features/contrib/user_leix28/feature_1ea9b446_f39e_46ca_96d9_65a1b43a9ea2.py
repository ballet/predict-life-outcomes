# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng.external import MinMaxScaler

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [x for x in list(X_df.columns) if "t5b1" in x]

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    return df.values.clip(1, 4).mean(axis=1)


# what is a brief name of this feature?
# type- str
name = "t5 social skills"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "t5 social skills, average score of t5b1"

# put it all together!
feature = Feature(input, transformer, name, description)
