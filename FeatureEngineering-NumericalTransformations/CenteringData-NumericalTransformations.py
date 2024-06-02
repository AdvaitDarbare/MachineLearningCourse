import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import codecademylib3_seaborn 

coffee = pd.read_csv('starbucks_customers.csv')

ages = coffee['age']

min_age = np.min(ages)
print(min_age)

max_age = np.max(ages)
print(max_age)

print(max_age - min_age)

mean_age = np.mean(ages)
print(mean_age)

centered_ages = ages - mean_age
centered_data = centered_ages
plt.hist(centered_data)
plt.show()

