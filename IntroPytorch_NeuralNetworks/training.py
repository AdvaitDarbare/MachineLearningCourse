<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# **Setup**
# 
# Run the following cell to import NumPy, pandas, and PyTorch.

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# **Import Data**
# 
# Run the code cell to import the dataset and select numerical features. In this exercise, we&#39;ll start using a larger set of numerical features:
# 
# - **bedrooms**: The number of bedrooms in the apartment
# - **bathrooms**: The number of bathrooms in the apartment
# - **size_sqft**: The size of the apartment in square feet  
# - **min_to_subway**: The number of minutes to the closest subway
# - **floor**: The building floor of the apartment
# - **building_age_yrs**: The age of the building in years
# - **no_fee**: Binary indicator that specifies whether the rental has a broker&#39;s fee (1) or not (0)
# - **has_roofdeck**: Binary indicator that specifies whether the rental has a roofdeck (1) or not (0)
# - **has_washer_dryer**: Binary indicator that specifies whether the rental has a washer/dryer units (1) or not (0)
# - **has_doorman**: Binary indicator that specifies whether the rental has a doorman (1) or not (0)
# - **has_elevator**: Binary indicator that specifies whether the rental has an elevator (1) or not (0)
# - **has_dishwasher**: Binary indicator that specifies whether the rental has a dishwasher (1) or not (0)
# - **has_patio**: Binary indicator that specifies whether the rental has a patio (1) or not (0)
# - **has_gym**: Binary indicator that specifies whether the rental has a gym (1) or not (0)
#  boroughs
# 
# 
# 

# In[2]:


apartments_df = pd.read_csv(&#34;streeteasy.csv&#34;)

numerical_features = [&#39;bedrooms&#39;, &#39;bathrooms&#39;, &#39;size_sqft&#39;, &#39;min_to_subway&#39;, &#39;floor&#39;, &#39;building_age_yrs&#39;,
                      &#39;no_fee&#39;, &#39;has_roofdeck&#39;, &#39;has_washer_dryer&#39;, &#39;has_doorman&#39;, &#39;has_elevator&#39;, &#39;has_dishwasher&#39;,
                      &#39;has_patio&#39;, &#39;has_gym&#39;]

# create tensor of input features
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
# create tensor of targets
y = torch.tensor(apartments_df[&#39;rent&#39;].values, dtype=torch.float).view(-1,1)


# **Checkpoint 1/3**
# 
# We&#39;ve created a neural network, defined a loss function, and initialized an optimizer already.
# 
# Note on the network: since we now have `14` input features, we&#39;ve increased the size of our hidden layers to try to capture that complexity. But we are still following a common neural network architecture where the first hidden layer has twice as many nodes (128) as the next (64).
# 
# We&#39;ve also tried to write a training loop to train our neural network for 500 epochs, but we&#39;ve made a mistake. As a result, our loss values are extremely high!
# 
# Can you spot our error and fix the training loop?
# 
# Don’t forget to run and save the cell before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## YOUR SOLUTION HERE ##
num_epochs = 500
for epoch in range(num_epochs):
    predictions = model(X) 
    MSE = loss(predictions, y)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f&#39;Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}&#39;)


# **Checkpoint 2/3**
# 
# Using the same model set up, we&#39;ve written another training loop. This time we&#39;ve made a different mistake, and the loss values don&#39;t seem to be changing from epoch to epoch.
# 
# Can you fix our mistake?
# 
# Don’t forget to run and save the cell before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## YOUR SOLUTION HERE ##
num_epochs = 500
for epoch in range(num_epochs):
    predictions = model(X) 
    MSE = loss(predictions, y) 
    MSE.backward()
    optimizer.step() 
    optimizer.zero_grad()
    
    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f&#39;Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}&#39;)


# **Checkpoint 3/3**
# 
# We&#39;ve already defined a neural network, loss function, and optimizer. 
# 
# Fill in a training loop that will train the neural network `model` for 1000 epochs.
# 
# Do the additional epochs improve the model performance?
# 
# Don’t forget to run and save the cell before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[5]:


# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## YOUR SOLUTION HERE ##
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X) # forward pass 
    MSE = loss(predictions, y) # calculate the loss 
    MSE.backward()
    optimizer.step() # update the weights and biases
    optimizer.zero_grad()
    
    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f&#39;Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}&#39;)


# In[ ]:




<script type="text/javascript" src="/relay.js"></script></body></html>
