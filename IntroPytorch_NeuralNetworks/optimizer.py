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

# **Import Libraries**
# 
# Run the cell below to import NumPy, pandas, and PyTorch.

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# **Checkpoint 1/3**
# 
# Import PyTorch&#39;s optimizers using the alias `optim`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


## YOUR SOLUTION HERE ##
import torch.optim as optim


# **Checkpoint 2/3**
# 
# We&#39;ve defined a sequential neural network `model` already in the Checkpoint 2 code cell.
# 
# Initialize the Adam optimizer using `model`&#39;s parameters and a **learning rate** set to `0.001`. 
# 
# Save the result to the variable `optimizer`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# set a random seed - do not modify
torch.manual_seed(42) 

# create neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

## YOUR SOLUTION HERE ## 
# learning rate is lr
optimizer = optim.Adam(model.parameters(),lr=0.001)


# **Checkpoint 3/3**
# 
# We&#39;ve already defined a neural network, run a forward pass with some of our apartment data, and computed the loss from that forward pass.
# 
# At `## YOUR SOLUTION HERE ##`, add code to
# 
# 1. initialize `Adam` in the variable `optimizer` with a learning rate of `.001`
# 2. use the loss we&#39;ve already assigned to `MSE` to perform the backwards pass
# 3. use `optimizer` to update the weights and biases
# 
# Run the notebook again. Did our optimization decrease the loss at all? 
# 
# After testing your work with the learning rate of `.001`, try some other learning rates to see how that impacts the optimization.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# set a random seed - do not modify
torch.manual_seed(42)

# create neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

# import the data
apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)
numerical_features = [&#39;bedrooms&#39;, &#39;bathrooms&#39;, &#39;size_sqft&#39;]
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
y = torch.tensor(apartments_df[&#39;rent&#39;].values,dtype=torch.float)

# forward pass
predictions = model(X)

# define the loss function and compute loss
loss = nn.MSELoss()
MSE = loss(predictions,y)
print(&#39;Initial loss is &#39; + str(MSE))

## YOUR SOLUTION HERE ##
optimizer = optim.Adam(model.parameters(), lr=0.001)
MSE.backward()
optimizer.step()

# feed the data through the updated model and compute the new loss
predictions = model(X)
MSE = loss(predictions,y)
print(&#39;After optimization, loss is &#39; + str(MSE))

</details><script type="text/javascript" src="/relay.js"></script></body></html>
