import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

# Store iris.data
samples = iris.data

# Create x and y
x = samples[:,0] # contains the column 0 values of samples
y = samples[:,1] # contains the column 1 values of samples

# Plot x and y
plt.scatter(x, y, alpha=0.5)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')


# Show the plot
plt.show()

# relationship between the sepal length (cm) and sepal width (cm)
