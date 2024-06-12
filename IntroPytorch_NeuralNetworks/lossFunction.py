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

# **Setup**
# 
# Run the following cell to import NumPy, pandas, and PyTorch.

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# **Checkpoint 1/3**
# 
# Suppose we fed two apartments through a neural network with the result
# 
# - predictions: 750, 1000
# - targets (actual rent): 1000, 900
# 
# We&#39;ve tried to calculate MSE below, but we&#39;ve made a mistake. Can you fix our calculation?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


## YOUR SOLUTION HERE ##
difference1 = 750 - 1000
difference2 = 1000 - 900
MSE = (difference1**2 + difference2**2)/2

# show output
MSE


# **Checkpoint 2/3**
# 
# In Exercise 7 we built the neural network
# 
# ```py
# model = nn.Sequential(
#     nn.Linear(3,16),
#     nn.ReLU(),
#     nn.Linear(16,8),
#     nn.ReLU(),
#     nn.Linear(8,4),
#     nn.ReLU(),
#     nn.Linear(4,1)
# )
# ```
# 
# When we fed our apartment data forward through the network, the first five predictions were 
# ```py
# predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float)
# ```
# 
# The first five targets (actual rent values) were
# 
# ```py
# y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)
# ```
# 
# Use PyTorch&#39;s `nn.MSELoss()` function to compute the MSE of these five apartments. Assign the result to the variable `MSE`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# define prediction and target tensors
predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float)
y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)

## YOUR SOLUTION HERE ##
loss = nn.MSELoss()
MSE = loss(predictions, y)

# show output
print(&#34;MSE Loss:&#34;, MSE)


# **Checkpoint 3/3**
# 
# Let&#39;s make the result from Checkpoint 2 a bit more interpretable.
# 
# Compute the square root of the `MSE` tensor from Checkpoint 2 (hint: `variable**(1/2)` will compute the square root of a tensor assigned to `variable`). Assign the result to the variable `RMSE` (this stands for **root** mean squared error).
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


## YOUR SOLUTION HERE ##
RMSE = MSE**(1/2)

# show output
RMSE


# Looks like this model&#39;s error for rent in dollars is around `$6200`. We can improve this quite a bit in training!
</details><script type="text/javascript" src="/relay.js"></script></body></html>
