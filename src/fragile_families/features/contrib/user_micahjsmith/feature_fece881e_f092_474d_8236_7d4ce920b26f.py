# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullFiller, ColumnSelector
import numpy as np

# what are the input columns to this feature?
# type - Union[str, List[str]]
input = ["m1a4", "cf1cohm", "f1b2", "f2c5"]


def clean(df):
    df = df.copy()
    df.replace(-9, np.nan, inplace=True)

    # clean m1a4
    df["m1a4"] = df["m1a4"].replace(2, 0).replace(3, 0).replace(-3, np.nan)

    df["f1b2"].replace(2, 0, inplace=True)
    df["f2c5"].replace(2, 0, inplace=True)
    df["f2c5"].replace(-6, 0, inplace=True)

    married_missing = df["m1a4"].isnull()
    df["m1a4"].where(df["m1a4"].notnull(), df["f1b2"], inplace=True)
    df["m1a4"].where(df["m1a4"].notnull(), df["f2c5"], inplace=True)

    # clean cf1cohm
    df["cf1cohm"].fillna(0, inplace=True)

    # replace for nonmarried
    df["m1a4"].where(df["m1a4"] == 1, df["cf1cohm"], inplace=True)

    return df


# what transformations do you want to apply to these specific input columns?
# type - function/lambda, transformer-like (object with fit and transform methods), or list of functions/transformers
transformer = [
    clean,
    NullFiller(),
    ColumnSelector("m1a4"),
]

# what is a brief name of this feature?
# type - str
name = "married or cohabiting at birth"

# what is a longer human-readable description for this feature? you can include more background on your calculations or thinking
# type - Optional[str]
description = """\
Whether the baby's parents are either married or cohabiting at the time of the baby's birth. Try to impute married by father survey. If cannot impute, then default to 0.
"""

# put it all together!
feature = Feature(input, transformer, name, description)
