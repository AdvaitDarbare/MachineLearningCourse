<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

'''
Linear Regression with Perceptrons
- Transform equation into neural network structure called a Perception
- Equation: rent = 2.5sqft -1.5age + 1000


Input layer					Output Layer

size_sqft  ------ 2.5     —>

Age  ----------- -1.5	—>			+                     

1  -------------- 1000   —>


- A Perceptron is a type of network structure consisting of nodes (circles in the diagram) connected to each other by edges (arrows in the diagram). The nodes appear in vertical layers, connected from left to right.


'''

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

# **Example: Linear Regression as a Perceptron**
# 
# Here&#39;s the network from the narrative example:
# ![A network diagram consisting of circles connected by arrows. From left to right, there are two vertical stacks of nodes called layers. The first, called the input layer, has three nodes, labelled size_sqft, age, and 1. Each of these three nodes has an arrow leading to the single node in the second layer labelled the output layer. This node has a plus sign in it. The edge leading from the square foot node to output has the number 2.5 on it. The edge leading from the age node to the output has the number -1.5 on it. The edge leading from the node labelled 1 to the output has the number 1000 on it.](https://static-assets.codecademy.com/Courses/intro-to-pytorch/network1.svg)
# 
# Let&#39;s use Python to perform the computation.

# In[1]:


# Define the inputs
size_sqft = 500.0
age = 10.0
bias = 1

# The inputs flow through the edges, receiving weights
weighted_size = 2.5 * size_sqft
weighted_age = -1.5 * age
weighted_bias = 1000 * bias

# The output node adds the weighted inputs
weighted_sum = weighted_size + weighted_age + weighted_bias

# Generate prediction
print(&#34;Predicted Rent:&#34;, weighted_sum)


# **Checkpoint 1 / 1**
# 
# Let&#39;s add an additional input feature `bedrooms` to our linear regression perceptron which corresponds to the number of bedrooms in the apartment.
# 
# Here&#39;s a new network incorporating `bedrooms`:
# 
# ![description below the image](https://static-assets.codecademy.com/Courses/intro-to-pytorch/network5b.svg)
# 
# This new network has a weight of `3` for the `size_sqft` input, a weight of `-2.3` for the `age` input, a weight of `100` for the `bedrooms` input, and a weight of `500` for the bias.
# 
# Using this updated network, predict the rent for an apartment with 
# - `size_sqft` equal to `1250.0`
# - `age` equal to `15.0` years
# - number of `bedrooms` equal to `2.0`
# 
# Save the predicted rent price to the variable `weighted_sum`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! See the `Jupyter Help` toggle at the top of the notebook for more details.

# In[6]:


## YOUR SOLUTION HERE ##

# Define the inputs
size_sqft = 1250.0
age = 15.0
bedrooms = 2.0
bias = 1

# The inputs flow through the edges, receiving weights
weighted_size = 3 * size_sqft
weighted_age = -2.3 * age
weighted_bedrooms = 100 * bedrooms
weighted_bias = 500 * bias

# The output node adds the weighted inputs
weighted_sum = weighted_size + weighted_age + weighted_bedrooms + weighted_bias

# Generate prediction
print(&#34;Predicted Rent:&#34;, weighted_sum)


# In[ ]:




</details><script type="text/javascript" src="/relay.js"></script></body></html>
