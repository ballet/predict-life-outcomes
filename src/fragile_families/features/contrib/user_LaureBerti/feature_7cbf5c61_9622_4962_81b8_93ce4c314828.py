# include any imports used in this feature right here (within this code cell)
from ballet import Feature
import numpy as np
import pandas as pd


# what are the input columns to this feature?
# type- Union[str, List[str]]
m_inc = ["cm1hhinc", "cm2hhinc", "cm3hhinc", "cm4hhinc", "cm5hhinc"]
f_inc = ["cf1hhinc", "cf2hhinc", "cf3hhinc", "cf4hhinc", "cf5hhinc"]

m_adult = ["cm1adult", "cm2adult", "cm3adult", "cm4adult", "cm5adult"]
f_adult = ["cf1adult", "cf2adult", "cf3adult", "cf4adult", "cf5adult"]

input = m_inc + f_inc + m_adult + f_adult

# what transformations do you want to apply to these specific input columns?
# type- function/lambda, transformer-like (object with fit and transform
#       methods), or list of functions/transformers
def transformer(df):
    df.replace(-9, np.nan)
    df["cm1hhinc"].fillna(df["cm1hhinc"].mode()[0], inplace=True)
    df["cm2hhinc"].fillna(df["cm2hhinc"].mode()[0], inplace=True)
    df["cm3hhinc"].fillna(df["cm3hhinc"].mode()[0], inplace=True)
    df["cm4hhinc"].fillna(df["cm4hhinc"].mode()[0], inplace=True)
    df["cm5hhinc"].fillna(df["cm5hhinc"].mode()[0], inplace=True)
    df["cf1hhinc"].fillna(df["cf1hhinc"].mode()[0], inplace=True)
    df["cf2hhinc"].fillna(df["cf2hhinc"].mode()[0], inplace=True)
    df["cf3hhinc"].fillna(df["cf3hhinc"].mode()[0], inplace=True)
    df["cf4hhinc"].fillna(df["cf4hhinc"].mode()[0], inplace=True)
    df["cf5hhinc"].fillna(df["cf5hhinc"].mode()[0], inplace=True)

    m_inc_avg = df[m_inc].clip(0, None).min(axis=1)
    f_inc_avg = df[f_inc].clip(0, None).min(axis=1)
    m_adult_avg = df[m_adult].clip(0, None).max(axis=1)
    f_adult_avg = df[f_adult].clip(0, None).max(axis=1)
    ratio = (f_inc_avg + m_inc_avg) / np.max([m_adult_avg, m_adult_avg])

    return np.array(pd.DataFrame({"ratio_muli_year": ratio}))


# what is a brief name of this feature?
# type- str
name = "Ratio of minimal income per adult"

# what is a longer human-readable description for this feature? you can include
# more background on your calculations or thinking
# type- Optional[str]
description = "The ratio of  minimal income per maximal number of adults in the household (multi-year over all the waves) "

# put it all together!
feature = Feature(input, transformer, name=name, description=description)
