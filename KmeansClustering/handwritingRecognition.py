import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# load the digits dataset and print it contents
digits = datasets.load_digits()
print(digits)

# print the description of the dataset
print(digits.DESCR)

# print the data of digits dataset
# each list contains 64 values which represent pixxrl colors of image 0-16, 0 is white and 16 is black
print(digits.data)

# print the target values of the digits dataset
print(digits.target)


plt.gray()

plt.matshow(digits.images[100])

plt.show()

print(digits.target[100])

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()


# K-Means Clustering

# cluster 1797 different digits iamges to groups

from sklearn.cluster import KMeans

# 10 clusters becaues there are 10 digits
# using random state in order to ensure that every time that we run the code, the model is built in the same way
model = KMeans(n_clusters=10, random_state=42)

# fit the digits data to the model
model.fit(digits.data)

# Visualize the centroids, data samples are on the 64-dimensional space

# add figure of size 8x3
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # Initialize subplots in a grid of 2x5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    
    # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.68,5.87,6.72,5.65,1.83,0.00,0.00,0.61,6.41,2.44,0.99,4.04,4.58,0.00,0.00,0.00,0.00,0.00,0.00,2.37,5.34,0.00,0.00,0.00,0.00,0.00,0.00,1.53,6.11,0.00,0.00,0.00,2.14,6.56,5.80,4.43,5.88,0.00,0.00,0.00,2.67,6.41,6.94,7.40,6.71,2.52,0.00,0.00,0.15,4.04,2.75,0.46,1.76,6.10,3.36],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.45,5.95,6.10,4.57,0.76,0.00,0.00,0.00,5.87,3.59,1.91,4.73,4.65,0.00,0.00,0.38,7.09,0.23,0.00,0.92,6.86,0.00,0.00,0.53,7.17,0.00,0.00,0.00,6.56,1.22,0.00,0.00,6.56,2.29,0.00,1.15,6.86,0.76,0.00,0.00,1.68,6.71,3.74,6.79,2.60,0.00,0.00,0.00,0.00,2.44,4.27,1.37,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.82,5.27,4.58,2.60,0.23,0.00,0.00,0.00,2.52,2.44,3.28,6.18,3.59,0.00,0.00,0.00,0.00,0.23,2.52,6.48,2.44,0.00,0.00,0.00,0.69,7.32,7.63,7.55,3.66,0.00,0.00,0.00,0.00,0.00,0.00,0.46,7.32,0.54,0.00,0.00,0.23,0.00,1.68,4.96,6.71,0.23,0.00,0.00,3.74,7.24,6.64,3.28,0.92,0.00,0.00],
[0.00,0.92,2.52,2.60,6.48,0.00,0.00,0.00,2.52,6.79,5.11,5.80,6.86,0.00,0.00,0.00,6.56,1.60,0.00,1.53,6.87,0.00,0.00,0.00,6.79,0.92,0.00,1.60,6.33,0.00,0.00,0.00,6.10,4.96,3.13,6.33,6.11,0.00,0.00,0.00,1.91,3.43,4.50,4.50,5.26,0.00,0.00,0.00,0.00,0.00,0.00,3.58,4.05,0.00,0.00,0.00,0.00,0.00,0.00,3.51,3.51,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

print(new_labels) 

# Assuming new_labels is an array containing the cluster labels assigned by k-means
for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')
