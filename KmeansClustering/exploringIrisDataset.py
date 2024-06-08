import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

# Iris dataset contains 3 different plant species, Iris setosa, Iris versicolor, Iris virginica

# each row of data is a sample, eacah flower is a sample

# features of datasets are:
# Column 0: Sepal length
# Column 1: Sepal width
# Column 2: Petal length
# Column 3: Petal width

# goal is to cluster the 3 species of iris plants

# load and print the iris dataset
iris = datasets.load_iris()
print(iris.data)

# target values, indicate which cluster each flower belong to, in real life datasets usually don't come up with targets

# ground truth is number coressponding to flower we are trying to learn

print(iris.target)

# single row of data and corresponding target
print(iris.data[0, :], iris.target[0])

# print the description of the dataset
print(iris.DESCR)
