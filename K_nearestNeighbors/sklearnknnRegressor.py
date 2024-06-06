# Importing the necessary modules
from movies import movie_dataset, movie_ratings  # Assuming movie_dataset and movie_ratings are predefined data

# Importing KNeighborsRegressor from sklearn
from sklearn.neighbors import KNeighborsRegressor

# Creating an instance of the KNeighborsRegressor with specified parameters
# n_neighbors=5 means the algorithm will consider the 5 nearest neighbors for making predictions
# weights="distance" means closer neighbors will have a greater influence on the prediction
regressor = KNeighborsRegressor(n_neighbors=5, weights="distance")

# Fitting the regressor to the movie dataset and corresponding ratings
regressor.fit(movie_dataset, movie_ratings)

# Predicting ratings for new data points
# The input data points represent features of new movies
predictions = regressor.predict([[0.016, 0.300, 1.022], [0.0004092981, 0.283, 1.0112], [0.00687649, 0.235, 1.0112]])

# Printing the predicted ratings
print(predictions)
