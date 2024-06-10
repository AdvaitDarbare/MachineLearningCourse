from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Makes concentric circles
points, labels = make_circles(n_samples=300, factor=.2, noise=.05, random_state = 1)

#Makes training set and validation set.
training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = "linear", random_state = 1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
print(training_data[0])



# Transform training data to 3D: For each point in training_data, calculate three new features:
# 1. The product of the point's coordinates multiplied by the square root of 2 (interaction term),
# 2. The square of the x-coordinate,
# 3. The square of the y-coordinate.
# This transformation projects the 2D data into a 3D space to potentially make it linearly separable.
new_training = [[2 ** 0.5 * pt[0] * pt[1], pt[0] ** 2, pt[1] ** 2] for pt in training_data]

# Transform validation data to 3D: Similar to the training data, for each point in validation_data,
# calculate the same three new features to project the validation data into the same 3D space.
# This ensures consistency in dimensions between training and validation datasets for the SVM classifier.
new_validation = [[2 ** 0.5 * pt[0] * pt[1], pt[0] ** 2, pt[1] ** 2] for pt in validation_data]


classifier.fit(new_training, training_labels)
print(classifier.score(new_validation, validation_labels))
