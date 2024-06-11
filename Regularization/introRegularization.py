'''
1. Why Regularize?
- minimizes overfitting, executed during the model fitting step
- embeded feature selection
- how well it can generalize from known to unknown data
- Regularization makes sure that our model is still accurate

2. What is overfitting?
- model is able to represent a particular set of points but not new data well is overfitting
- It fits the training data well but performs significantly worse on test data
- It typically has more parameters than necessary, i.e., it has high model complexity
- It might be fitting for features that are multi-collinear (i.e., features that are highly negatively or positively correlated)
- It might be fitting the noise in the data and likely mistaking the noise for features
- For instance if the R-squared score is high for training data but the model performs poorly on test data, it’s a strong indicator of overfitting.
'''

import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt

df = pd.read_csv("./student_math.csv")
print(df.head())

#setting target and predictor variables
y = df['Final_Grade']
X = df.drop(columns = ['Final_Grade'])

# 1. Number of features
num_features = len(X.columns)
print("Number of features: ",num_features)

#Performing a Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Fitting a Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Training Error
pred_train = model.predict(X_train)
MSE_train = np.mean((pred_train - y_train)**2)
print("Training Error: ", MSE_train)

# 2. Testing Error
pred_test = model.predict(X_test)
MSE_test = np.mean((pred_test - y_test)**2)
print("Testing Error: ", MSE_test)

#Calculating the regression coefficients
predictors = X.columns
coef = pd.Series(model.coef_,predictors).sort_values()

# 3. Plotting the Coefficients

plt.figure(figsize = (15,10))
coef.plot(kind='bar', fontsize = 20)
plt.title ("Regression Coefficients", fontsize = 30)
plt.show()

# error on the test data is almost double the error with the training data means overfitting


