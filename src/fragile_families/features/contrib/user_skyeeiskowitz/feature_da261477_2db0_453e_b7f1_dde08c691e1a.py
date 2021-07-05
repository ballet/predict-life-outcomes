import ballet
from ballet import Feature
from ballet.eng.external import SimpleImputer
import numpy as np

# Collection of columns that indicate mother and father drug involvment in wave 3
input = [
    "f3j44j",
    "f3j44i",
    "f3i28_3",
    "f3j45",
    "f3j46",
    "f3j47",
    "f3j48",
    "f3j49",
    "f3j50",
    "f3j51",
    "f3j55a",
    "m3i28_3",
    "m3j36j",
    "m3j37",
    "m3j38",
    "m3j40",
    "m3j43",
    "m3j52a",
    "m3j52b",
    "m3j53",
]

# The value replacer replaces positive entries that do not indicate a 'yes' response to drug involvement with 0
# Then, negative values are replaced with NAN to be imputed later
# The columns are then summed as a count to yes type responses to drug involvement
transformer = [
    ballet.eng.ValueReplacer(2, 0),
    lambda df: df.where(df >= 0, np.nan),
    SimpleImputer(strategy="constant", fill_value=0),
    lambda df: df.sum(axis=1),
]

name = "Parent drug history"

description = "Indication of drug involvement of the mother and father in wave 3  by counting relevant yes responses to drug involvement questions."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
