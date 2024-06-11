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

'''
3. Loss function
- loss function is minimzed by the OLS, larger data sets the gradient descent

4. Regularization
- penalizes models for overfitting by adding a “penalty term
- New loss function=Old loss function+α∗Regularization term

5. L1 or Lasso Regularization
- Loss=n1​∑i=1n​(yi​−(b0​+b1​x1i​+b2​x2i​))2
- L1 Loss=Loss+α⋅(∣b1∣+∣b2∣)
- reduce the importance (coefficients) of one feature to zero if it finds that feature is not useful for predicting results in a simpler model that generalizes better to new data.
- 

'''

import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import helpers


df = pd.read_csv("./student_math.csv")
y = df['Final_Grade']
X = df.drop(columns = ['Final_Grade'])

# 1. Train-test split and fitting an l1-regularized regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)

l1_pred_train = lasso.predict(X_train)
l1_mse_train = np.mean((l1_pred_train - y_train)**2)
print("Lasso (L1) Training Error: ", l1_mse_train)

# 2. Calculate testing error
l1_pred_test = lasso.predict(X_test)
l1_mse_test = np.mean((l1_pred_test - y_test)**2)
print("Lasso (L1) Testing Error: ", l1_mse_test)

# 3. Plotting the Coefficients
predictors = X.columns
coef = pd.Series(lasso.coef_,predictors).sort_values()
plt.figure(figsize = (12,8))
plt.ylim(-1.0,1.0)
coef.plot(kind='bar', title='Regression Coefficients with Lasso (L1) Regularization')
plt.show()


'''
6. L2 Ridge
- L2 Loss=Original Loss+α⋅(b1^2+b2^2)
- it keeps all features in the model, just with smaller coefficients rather than eliminating like L1,  L2 does not set coefficients to zero.
'''
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import helpers

df = pd.read_csv("./student_math.csv")
y = df['Final_Grade']
X = df.drop(columns = ['Final_Grade'])

# 1. Train-test split and fitting an l2-regularized regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ridge = Ridge(alpha = 100)
ridge.fit(X_train, y_train)

#Training error
l2_pred_train = ridge.predict(X_train)
l2_mse_train = np.mean((l2_pred_train - y_train)**2)
print("Ridge (L2) Training Error: ", l2_mse_train)

# 2. Calculate testing error
l2_pred_test = None
l2_mse_test = None
print("Ridge (L2) Testing Error: ", l2_mse_test)


# 3. Plotting the Coefficients
predictors = X.columns
coef = pd.Series(ridge.coef_,predictors).sort_values()
plt.figure(figsize = (12,8))
plt.ylim(-1.0,1.0)
coef.plot(kind='bar', title='Regression Coefficients with Lasso (L1) Regularization')



