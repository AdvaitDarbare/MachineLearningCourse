# Import necessary libraries
import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function

# Load the dataset from the URL
streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

# Convert the loaded data into a DataFrame
df = pd.DataFrame(streeteasy)

# Select the features for the model
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

# Select the target variable
y = df[['rent']]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state = 6)

# Print the shapes of the training and testing sets to verify the split
print(x_train.shape)  # Print the shape of x_train
print(x_test.shape)   # Print the shape of x_test
print(y_train.shape)  # Print the shape of y_train
print(y_test.shape)   # Print the shape of y_test
