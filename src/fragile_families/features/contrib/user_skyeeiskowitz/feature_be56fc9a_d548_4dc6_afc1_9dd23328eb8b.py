import ballet
from ballet import Feature
from ballet.eng.external import SimpleImputer

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

# The value replacer replaces all entries that do not indicate a 'yes' response to drug involvement with 0
tansformer = [
    ballet.eng.ValueReplacer(-9, 0),
    ballet.eng.ValueReplacer(-6, 0),
    ballet.eng.ValueReplacer(-3, 0),
    ballet.eng.ValueReplacer(-2, 0),
    ballet.eng.ValueReplacer(-1, 0),
    ballet.eng.ValueReplacer(2, 0),
    lambda df: df["f3j44j"]
    + df["f3j44i"]
    + df["f3i28_3"]
    + df["f3j45"]
    + df["f3j46"]
    + df["f3j47"]
    + df["f3j48"]
    + df["f3j49"]
    + df["f3j50"]
    + df["f3j51"]
    + df["f3j55a"]
    + df["m3i28_3"]
    + df["m3j36j"]
    + df["m3j37"]
    + df["m3j38"]
    + df["m3j40"]
    + df["m3j43"]
    + df["m3j52a"]
    + df["m3j52b"]
    + df["m3j53"],
    SimpleImputer(strategy="median"),
]


name = "Parent drug history"

description = "Indication of drug involvement of the mother and father in wave 3  by counting relevant yes responses to drug involvement questions."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
