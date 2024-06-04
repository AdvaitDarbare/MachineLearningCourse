# Import necessary libraries
import codecademylib3_seaborn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Define temperature array
temperature = np.array(range(60, 100, 2))
# Reshape temperature array for use in linear regression
temperature = temperature.reshape(-1, 1)

# Define sales data
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

# Plot temperature vs sales data
plt.plot(temperature, sales, 'o')

# Initialize Linear Regression model
line_fitter = LinearRegression()

# Fit the model using temperature and sales data
line_fitter.fit(temperature, sales)

# Predict sales based on the fitted model
sales_predict = line_fitter.predict(temperature)

# Plot the predicted sales against temperature
plt.plot(temperature, sales_predict)

# Display the plot
plt.show()
