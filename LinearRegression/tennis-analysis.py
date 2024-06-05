import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv");
print(df.head())
print(df.columns)
print(df.describe())

# Perform exploratory analysis
# Plotting different features against the different outcomes
print(df.corr())
plt.scatter(df['TotalPointsWon'], df['Winnings'])
plt.title('TotalPointsWon vs Winnings')
plt.xlabel('TotalPointsWon')
plt.ylabel('Winnings')
plt.show()
plt.clf()


plt.scatter(df['Aces'], df['Winnings'])
plt.title('Aces vs Winnings')
plt.xlabel('Aces')
plt.ylabel('Winnings')
plt.show()
plt.clf()


plt.scatter(df['FirstServePointsWon'], df['Winnings'])
plt.title('FirstServePointsWon vs Winnings')
plt.xlabel('FirstServePointsWon')
plt.ylabel('Winnings')
plt.show()
plt.clf()


plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()



# single feature LR model

features = df[['BreakPointsOpportunities']]
winnings = df[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)


print('Predicting Winnings with BreakPointsOpportunities Test Score:', model.score(features_test,winnings_test))

# 84.42% of the variance in Winnings can be explained by the BreakPointsOpportunities in the test dataset.

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


# Multiple Linear Regression

# select features and value to predict
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]
winnings = df[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with Multiple Features Test Score:', model.score(features_test,winnings_test))

# 81.45% of the variance in Winnings can be explained by the features in the test dataset

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()
