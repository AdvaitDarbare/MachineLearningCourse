import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd

# Load the Iris dataset
iris = datasets.load_iris()
samples = iris.data
target = iris.target
species = [iris.target_names[t] for t in target]

# Create a KMeans model with 3 clusters
model = KMeans(n_clusters=3)

# Fit the model to the data
model.fit(samples)

# Predict cluster labels for each data point
labels = [iris.target_names[s] for s in model.predict(samples)]

# Create a DataFrame with the predicted labels and actual species
df = pd.DataFrame({'labels': labels, 'species': species})

print(df)

# Create a crosstab to compare the cluster labels with the actual species
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
