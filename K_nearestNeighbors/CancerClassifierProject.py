import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Print the first data point
print(breast_cancer_data.data[0])
# Print feature names
print(breast_cancer_data.feature_names)

# Print target values
print(breast_cancer_data.target)
# Print target names
print(breast_cancer_data.target_names)

# Split dataset into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

# Print lengths of training data and labels
print(len(training_data))
print(len(training_labels))

# Calculate validation accuracies for different values of k
accuracies = []
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

# Generate list of k values
k_list = range(1, 101)
# Print accuracies
print(accuracies)

# Plot k vs. validation accuracies
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
