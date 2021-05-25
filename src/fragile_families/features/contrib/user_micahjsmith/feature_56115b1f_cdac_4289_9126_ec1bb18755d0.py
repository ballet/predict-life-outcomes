# include any imports used in this feature right here (within this code cell)
from ballet import Feature
from ballet.eng import NullFiller, ValueReplacer
import numpy as np

# what are the input columns to this feature?
input = ["m1a4", "cf1cohm", "f1b2", "f2c5"]

# what transformations do you want to apply to these specific input columns?
transformer = [
    ValueReplacer(-9, np.nan),
    # clean m1a4
    ("m1a4", [ValueReplacer(2, 0), ValueReplacer(3, 0), ValueReplacer(-3, np.nan),]),
    # clean f1b2
    ("f1b2", ValueReplacer(2, 0)),
    # clean f2c5
    ("f2c5", [ValueReplacer(2, 0), ValueReplacer(-6, 0),]),
    # replace m1a4 with f1b2 if it is missing
    (
        ["m1a4", "f1b2"],
        lambda df: df["m1a4"].where(df["m1a4"].notnull(), df["f1b2"]),
        "m1a4",
    ),
    # replace m1a4 with f2c5 it is is still missing
    (
        ["m1a4", "f2c5"],
        lambda df: df["m1a4"].where(df["m1a4"].notnull(), df["f2c5"]),
        "m1a4",
    ),
    # clean cf1cohm
    ("cf1cohm", NullFiller()),
    # replace for nonmarried
    (["m1a4", "cf1cohm"], lambda df: df["m1a4"].where(df["m1a4"] == 1, df["cf1cohm"])),
    # fill any remaining missing values with 0
    NullFiller(),
]

# what is a brief name of this feature?
name = "married or cohabiting at birth"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
description = """\
Whether the baby's parents are either married or cohabiting at the time of the
baby's birth. Try to impute married by father survey. If cannot impute, then
default to 0.
"""

# put it all together!
feature = Feature(input, transformer, name, description)
