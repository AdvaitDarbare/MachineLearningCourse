# Import the dataset and labels from the movies module
from movies import movie_dataset, labels

# Import the KNeighborsClassifier from the sklearn.neighbors module
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the KNeighborsClassifier with 5 neighbors
classifier = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier on the movie dataset and corresponding labels
classifier.fit(movie_dataset, labels)

# Predict the class labels for the new data points
# The new data points are [[.45, .2, .5], [.25, .8, .9], [.1, .1, .9]]
guess = classifier.predict([[.45, .2, .5], [.25, .8, .9], [.1, .1, .9]])

# Print the predicted class labels for the new data points
print(guess)
