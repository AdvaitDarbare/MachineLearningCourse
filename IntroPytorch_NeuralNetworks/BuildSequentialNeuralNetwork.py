<html><head></head><body>#!/usr/bin/env python


'''
Building sqeuential neural network, this did no training so the predicted values does not make sense
'''

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
# Run the cell below to import NumPy, pandas, and PyTorch.

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# #### Checkpoint 1/3

# Use `nn.Sequential` to create a neural network model with
# 
# - **input layer**: three nodes
# - **hidden layer**: eight nodes, with ReLU activation
# - **output layer**: one node
# 
# Assign your network to the variable `model`.
# 
# Because `nn.Sequential` will randomly initialize weights and biases, we&#39;ve set a **random seed** in every Checkpoint code cell. This is generally a good practice that means the initial weights and biases are the same every time we run the same code. Do not edit the seed value `42` as that is needed for our testing code.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


# set a random seed - do not modify
torch.manual_seed(42)

## YOUR SOLUTION HERE ##
model = nn.Sequential(
    nn.Linear(3,8),
    nn.ReLU(),
    nn.Linear(8,1)
)

# show model details
model


# <details><summary style="display:list-item; font-size:16px; color:blue;">Breakdown of this output</summary>
#     
# The output lists the sequential structure of the network. For example, the line
#     
# `(0): Linear(in_features=3, out_features=8, bias=True)`
#     
# indicates that we begin with a linear calculation from three nodes (the `in_features`) to eight nodes (the `out_features`). The line `bias=True` indicates that we are including a bias term (even though we aren&#39;t explicitly creating a bias node).

# #### Checkpoint 2/3

# Re-create the model from the prior checkpoint. This time, add a second hidden layer with
# 
# - four nodes
# - `nn.Sigmoid` as the activation function
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# set a random seed - do not modify
torch.manual_seed(42)

## YOUR SOLUTION HERE ##
model = nn.Sequential(
    nn.Linear(3,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.Sigmoid(),
    nn.Linear(4,1)
)

# show model details
model


# #### Dataset Import

# Let&#39;s create a model and feedforward all of our real Streeteasy data.
# 
# First, run the next code cell to import the Streeteasy data and convert the `size_sqft`, `bedrooms`, and `building_age_yrs` columns into a PyTorch tensor.
# 
# We will use these three columns to try to predict rent!

# In[4]:


# load pandas DataFrame
apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)

# create a numpy array of the numeric columns
apartments_numpy = apartments_df[[&#39;size_sqft&#39;, &#39;bedrooms&#39;, &#39;building_age_yrs&#39;]].values

# convert to an input tensor
X = torch.tensor(apartments_numpy,dtype=torch.float32)

# preview the first five apartments
X[:5]


# **Checkpoint 3/3**
# 
# Now that we have our data stored in `X`, let&#39;s create a neural network.
# 
# In the Checkpoint 3 code cell, we&#39;ve already created a neural network model. It follows a fairly common neural network architecture, where each hidden layer is halved in size as we feed forward to the output.
# 
# Run the feedforward process of `model` on `X`. Assign the result to the variable `predicted_rents`.
# 
# We&#39;ve added code to preview the first five predicted rents. What is the predicted rent for the first apartment in the dataset?
# 
# <details><summary style="display:list-item; font-size:16px; color:blue;">Our answer</summary>
# The first element of the output is -6.9229, so that is the prediction for the first apartment. The second apartment&#39;s predicted rent is -29.8163.</details>
# 
# Of course, these initial predictions will not be accurate at all. Without training, the feedforward is just applying randomly generated weights and biases! We&#39;ll train these on the actual rent values in a later exercise.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[5]:


# set a random seed - do not modify
torch.manual_seed(42)

# define the neural network
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
predicted_rent = model(X)

# show output
predicted_rent[:5]


# In[ ]:




</details></details><script type="text/javascript" src="/relay.js"></script></body></html>
