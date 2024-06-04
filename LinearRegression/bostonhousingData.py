import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load Boston housing dataset
boston = load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)

# Set the x-values to the nitrogen oxide concentration:
X = df[['NOX']]
# Y-values are the prices:
y = boston.target

# Initialize Linear Regression model
line_fitter = LinearRegression()
# Fit the model using NOX and target data
line_fitter.fit(X, y)

# Predict house prices based on the fitted model
ypred = line_fitter.predict(X)

# Plot the predicted prices against NOX
plt.plot(X, ypred)

# Display the plot
plt.show()

# Create a scatter plot of NOX vs house prices
plt.scatter(X, y, alpha=0.4)

# Plot the regression line
plt.plot(X, ypred, color='red')

#
