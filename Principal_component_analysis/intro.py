import pandas as pd
import codecademylib3

# Read the csv data as a DataFrame
df = pd.read_csv('./Dry_Bean.csv')

# Remove null and na values
df.dropna()

# 1. Print the DataFrame head
print(df.head())

# 2. Extract the numerical columns
data_matrix = df.drop(columns='Class')

# Extract the classes
classes = df['Class']

print(data_matrix)
