# include any imports used in this feature right here (within this code cell)

from ballet.eng.external import SimpleImputer
from ballet import Feature

# what are the input columns to this feature?
# type- Union[str, List[str]]
input = [
    "cf2gad_case",
    "cf2md_case_con",
    "cf2md_case_lib",
    "cm2gad_case",
    "cm2md_case_con",
    "cm2md_case_lib",
    "cf3alc_case",
    "cf3drug_case",
    "cf3gad_case",
    "cf3md_case_con",
    "cf3md_case_lib",
    "cm3alc_case",
    "cm3drug_case",
    "cm3gad_case",
    "cm3md_case_con",
    "cm3md_case_lib",
    "cf4md_case_con",
    "cf4md_case_lib",
    "cm4md_case_con",
    "cm4md_case_lib",
    "cm5md_case_con",
    "cm5md_case_lib",
    "cf5md_case_con",
    "cf5md_case_lib",
    "cn5md_case_con",
    "cn5md_case_lib",
]


# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
transformer = [
    lambda df: df[
        [
            "cf2gad_case",
            "cf2md_case_con",
            "cf2md_case_lib",
            "cm2gad_case",
            "cm2md_case_con",
            "cm2md_case_lib",
            "cf3alc_case",
            "cf3drug_case",
            "cf3gad_case",
            "cf3md_case_con",
            "cf3md_case_lib",
            "cm3alc_case",
            "cm3drug_case",
            "cm3gad_case",
            "cm3md_case_con",
            "cm3md_case_lib",
            "cf4md_case_con",
            "cf4md_case_lib",
            "cm4md_case_con",
            "cm4md_case_lib",
            "cm5md_case_con",
            "cm5md_case_lib",
            "cf5md_case_con",
            "cf5md_case_lib",
            "cn5md_case_con",
            "cn5md_case_lib",
        ]
    ].max(axis=1),
    SimpleImputer(strategy="most_frequent"),
]

# what is a brief name of this feature?
# type- str
name = "CIDI (alcohol, drug, depression, anxiety"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "Father or Mother with CIDI case."

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
