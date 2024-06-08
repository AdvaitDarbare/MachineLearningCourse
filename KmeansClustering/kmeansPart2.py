# Assign data samples to the nearest centroid.

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Step 2: Assign samples to nearest centroid

# Distance formula
def distance(a, b):
  return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5

# Cluster labels for each point (either 0, 1, or 2)

# create array called labels that assigns 0 to each datapoint
labels = np.zeros(len(samples))

# A function that assigns the nearest centroid to a sample
def assign_to_centroid(sample, centroids):
  k = len(centroids) # 3 centroids
  # distances array contain [0.0, 0.0, 0.0]
  distances = np.zeros(k)
  # iterate through number of centroids
  for i in range(k):
    # for each sample data point find the distance between data point and each centroid coordinate
    distances[i] = distance(sample, centroids[i])

  # retrieve the closest distance between data point and each centroid coordinate
  closest_centroid = np.argmin(distances)
  return closest_centroid

# Assign the nearest centroid to each sample
for i in range(len(samples)):
  labels[i] = assign_to_centroid(samples[i], centroids)

# Print labels
print(labels)
