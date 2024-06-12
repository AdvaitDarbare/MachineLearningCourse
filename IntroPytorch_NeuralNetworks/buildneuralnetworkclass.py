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


# **Checkpoint 1/3: Create a Neural Network in PyTorch Using OOP**
# 
# In the Checkpoint 1 code cell we&#39;ve included code to
# 
# 1. create the `NN_Regression` class from the narrative
# 2. create an input tensor of apartment data
# 3. feedforward to make predictions
# 
# Run the cell. The output from this network should be the same as the Sequential network from the last exercise! Once again, this network is completely untrained, so it isn&#39;t surprising that the &#34;predictions&#34; are very bad.
# 
# Note: even though you haven&#39;t written any code in this cell, still save and select `Test Work` to move on!
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class NN_Regression(nn.Module):
    def __init__(self):
        super(NN_Regression, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8) 
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1) 
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

## create an instance of NN_Regression
model = NN_Regression()

## create an input tensor

apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)
numerical_features = [&#39;size_sqft&#39;, &#39;bedrooms&#39;, &#39;building_age_yrs&#39;]
apartments_tensor = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)

## feedforward to predict rent
predicted_rents = model(apartments_tensor)

## show output
predicted_rents[:5]


# **Checkpoint 2/3**
# 
# We&#39;ve created a new neural network class called `OneHidden` with:
# 
# - a two node input layer
# - a four node hidden layer
# - a one node output layer
# 
# This class does not yet have any instructions for feedforward. Try running the cell without making any changes to the code. Without any linear layers or activation functions, the input tensor should be output as is.
# 
# In the `forward` method, define a feedforward that takes the input `x` and feeds it through:
# 
# 1. `self.layer1`
# 2. `self.relu`
# 3. `self.layer2`
# 
# Run the cell again to see the new predictions!
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class OneHidden(nn.Module):
    def __init__(self):
        super(OneHidden, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        ## YOUR SOLUTION HERE ##
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

## do not modify below this comment

# create an instance
model = OneHidden()

# create an input tensor
input_tensor = torch.tensor([3,4.5], dtype=torch.float32)

# run feedforward
predictions = model(input_tensor)

# show output
predictions


# **Checkpoint 3 / 3: Make a prediction**
# 
# Let&#39;s look at one way defining classes can be helpful.
# 
# We&#39;ve included the same `OneHidden` class from Checkpoint 2 in the Checkpoint 3 code cell below. But we&#39;ve updated the line
# 
# ```py
# def __init(self):
# ```
# 
# to have another input:
# 
# ```py
# def __init__(self,numHiddenNodes):
# ```
# 
# This input, `numHiddenNodes`, determines the number of hidden nodes in the hidden layer. For example, within the `__init__` method, the line
# 
# ```py
# self.layer1 = nn.Linear(2, numHiddenNodes)
# ```
# 
# uses `numHiddenNodes` to define the connections between the input and hidden layer.
# 
# Below `## YOUR SOLUTION HERE ##`, add the input `4` to the line `model = OneHidden` and run the cell. The output should be the same as for Checkpoint 2, where we hand coded the value `4` for the hidden layer.
# 
# Now, change the number of nodes in the hidden layer to `10` and run the cell again. How does the output change?
# 
# Save the notebook after running with `10` hidden layer nodes and test your work.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# set a random seed - do not modify
torch.manual_seed(42)

## create the NN_Regression class
class OneHidden(nn.Module):
    # add a new numHiddenNodes input
    def __init__(self, numHiddenNodes):
        super(OneHidden, self).__init__()
        # initialize layers
        # 3 input features, variable output features
        self.layer1 = nn.Linear(2, numHiddenNodes) 
        # variable input features, 8 output features
        self.layer2 = nn.Linear(numHiddenNodes, 1) 
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        ## YOUR SOLUTION HERE ##
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

## YOUR SOLUTION HERE ##
model = OneHidden(10)

## do not modify below this comment

# create an input tensor
input_tensor = torch.tensor([3,4.5], dtype=torch.float32)

# run feedforward
predictions = model(input_tensor)

# show output
predictions


# Pretty cool, right? This is just scratching the surface of the power of defining our own neural network classes. But in the next exercise, we&#39;ll return to the problem of actually improving our predictions by training sequential neural networks.
</details><script type="text/javascript" src="/relay.js"></script></body></html>
