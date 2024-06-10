# Predict Baseball Strike Zones With Machine Learning

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# print the features of aarons judge
print(aaron_judge.columns)

print('------------------------------')

# see the different values the description feature could have
print(aaron_judge.description.unique())

print('------------------------------')

# pitch was either a ball or a strike type feature; Every pitch is either a 'S', a 'B', or an 'X'.
print(aaron_judge.type.unique())

print('------------------------------')

# mapping value of S to 1 and B to 0
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})

print(aaron_judge['type'])

print('------------------------------')

# plotting the pitches

# how left or right pitch was from center of plate
print(aaron_judge['plate_x'])

print('------------------------------')

# drop the NaN values
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])

# plot values
plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.5)


# building the svm
training_set, validation_set = train_test_split(aaron_judge, random_state=1)

classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)

classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

draw_boundary(ax, classifier)

plt.show()

# Optimizing the SVM

# print the score
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
