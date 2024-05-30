import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

ca_housing = "https://raw.githubusercontent.com/csbfx/advpy122-data/master/California_housing.csv"

housing = pd.read_csv(ca_housing)
housing.info()

housing.head()

housing["ocean_proximity"].value_counts()

housing.describe()

housing.hist(bins=50, figsize=(20,15))
plt.show()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist();

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_train_set
