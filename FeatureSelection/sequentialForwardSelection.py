# builds feature by stating wiht none and then adding one feature at a time until desired

#first train and test using one feature
# each subsequent step, test modle on each new possible feature addition

# fourth feature to the feature set {age, weight, resting_heart_rate}

# features to choose from: age, height, weight, blood_pressure, and resting_heart_rate


set1 = {"age", "height", "weight", "resting_heart_rate"}

set2 = {"age", "weight", "blood_pressure", "resting_heart_rate"}


########

# Sequential Forward Selection with mlxtend

import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward selection
sfs = SFS(lr,
           k_features=3, # number of features to select
           forward=True,
           floating=False,
           scoring='accuracy',
           cv=0)
# Fit the equential forward selection model
sfs.fit(X, y)


#########

# Evaluating the Result of Sequential Forward Selection

import pandas as pd
import codecademylib3
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward selection
sfs = SFS(lr,
          k_features=3,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=0)
sfs.fit(X, y)

# Print the chosen feature names
print(sfs.subsets_[3]['feature_names'])

# Print the accuracy of the model after sequential forward selection
print(sfs.subsets_[3]['avg_score'])

# Plot the model accuracy
plot_sfs(sfs.get_metric_dict())
plt.show()

