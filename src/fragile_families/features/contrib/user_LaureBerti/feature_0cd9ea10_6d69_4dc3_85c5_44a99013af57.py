# include any imports used in this feature right here (within this code cell)

from ballet import Feature
import numpy as np
import pandas as pd
from ballet.eng.external import SimpleImputer

# what are the input columns to this feature?
# type- Union[str, List[str]]
m_inc = ["cm1hhinc", "cm2hhinc", "cm3hhinc", "cm4hhinc", "cm5hhinc"]
f_inc = ["cf1hhinc", "cf2hhinc", "cf3hhinc", "cf4hhinc", "cf5hhinc"]

m_adult = ["cm1adult", "cm2adult", "cm3adult", "cm4adult", "cm5adult"]
f_adult = ["cf1adult", "cf2adult", "cf3adult", "cf4adult", "cf5adult"]

input = m_inc + f_inc + m_adult + f_adult


def compute_ratio(df):
    print(df)
    df = pd.DataFrame(df, columns=[m_inc + f_inc + m_adult + f_adult])
    m_inc_avg = df[m_inc].clip(0, None).min(axis=1)
    f_inc_avg = df[f_inc].clip(0, None).min(axis=1)
    m_adult_avg = df[m_adult].clip(0, None).max(axis=1)
    f_adult_avg = df[f_adult].clip(0, None).max(axis=1)
    max_adult = np.max([m_adult_avg, f_adult_avg])
    ratio = (f_inc_avg + m_inc_avg) / max_adult
    return ratio


transformer = [
    lambda df: df.replace(-9, np.nan),
    SimpleImputer(strategy="mean"),
    compute_ratio,
]


# what is a brief name of this feature?
# type- str
name = "Income per adult ratio"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The minimal income ratio per maximal number of adults in the household (multi-year and annual for all the waves) "

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
