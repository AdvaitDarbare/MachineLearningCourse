<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# <details><summary style="display:list-item; font-size:16px; color:blue;">Jupyter Help</summary>
#     
# Having trouble testing your work? Double-check that you have followed the steps below to write, run, save, and test your code!
#     
# [Click here for a walkthrough GIF of the steps below](https://static-assets.codecademy.com/Courses/ds-python/jupyter-help.gif)
# 
# Run all initial cells to import libraries and datasets. Then follow these steps for each question:
#     
# 1. Add your solution to the cell with `## YOUR SOLUTION HERE ## `.
# 2. Run the cell by selecting the `Run` button or the `Shift`+`Enter` keys.
# 3. Save your work by selecting the `Save` button, the `command`+`s` keys (Mac), or `control`+`s` keys (Windows).
# 4. Select the `Test Work` button at the bottom left to test your work.
# 
# ![Screenshot of the buttons at the top of a Jupyter Notebook. The Run and Save buttons are highlighted](https://static-assets.codecademy.com/Paths/ds-python/jupyter-buttons.png)

# #### Checkpoint 1/3

# Follow the Checkpoint 1 instructions (below the exercise text to the left) to run the following code cell, save the notebook, and run our code tests.

# In[1]:


# do not modify this code
import pandas as pd
import torch
import numpy as np


# #### Checkpoint 2/3
# 
# Create a tensor containing the values `[2000,500, 7]` with the `torch.int` data type. Assign your tensor to the variable `apartment_tensor`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


## YOUR SOLUTION HERE ##

apartment_tensor = np.array([2000,500, 7])

# show output
apartment_tensor = torch.tensor(apartment_tensor, dtype=torch.int)

print(apartment_tensor)


# #### Checkpoint 3/3
# 
# In this checkpoint&#39;s code cell, we&#39;ve already added code to create a DataFrame named `apartments_df`. This array includes data on rent, size, and age of the apartments in our dataset.
# 
# Use `torch.tensor()` to convert this DataFrame to a tensor, with `torch.float32` as the datatype. Assign the result to the variable `apartments_tensor`.
# 
# Add your solution to the code cell directly below the comment `## YOUR SOLUTION HERE ##`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[ ]:


# import the dataset using pandas
apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)

# select the rent, size, and age columns
apartments_df = apartments_df[[&#34;rent&#34;, &#34;size_sqft&#34;, &#34;building_age_yrs&#34;]]

## YOUR SOLUTION HERE ##
apartments_tensor = torch.tensor(apartments_df.values, dtype=torch.float32)

# show output
print(apartments_tensor)

</details><script type="text/javascript" src="/relay.js"></script></body></html>