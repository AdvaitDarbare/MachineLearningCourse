# Import necessary libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load the honey production data
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Display the first few rows of the dataframe
print(df.head())

# Group the data by year and calculate the mean total production for each year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Display the yearly average production
print(prod_per_year)

# Reshape the year data to fit the model
X = prod_per_year['year'].values.reshape(-1,1)

# Display the reshaped data
print(X)

# Assign the total production data to Y
Y = prod_per_year['totalprod']

# Display the total production data
print(Y)

# Plot the data
plt.scatter(X,Y)
plt.show()

# Create a linear regression model
regr = linear_model.LinearRegression()

# Fit the model with the data
regr.fit(X,Y)

# Display the coefficient and intercept of the model
print(regr.coef_[0])
print(regr.intercept_)

# Predict the total production for the given years
y_predict = regr.predict(X)

# Plot the predicted data
plt.plot(X, y_predict)
plt.show()

# Create an array for future years
X_future = np.array(range(2013, 2050)).reshape(-1,1)

# Predict the total production for the future years
future_predict = regr.predict(X_future)

# Plot the future predictions
plt.plot(X_future, future_predict)
plt.show()
