<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

'''
Activation functions
* One of the ways neural networks move beyond linear regression is by incorporating non-linear activation functions. These functions allow a neural network to model nonlinear relationships within a dataset, which are very common and cannot be modeled with linear regression.

The process with an activation function is
* 		receive the weighted inputs
* 		add them up (to produce the same linear equation as before)
* 		apply an activation function

ReLU Activation Function

- If a number is negative, ReLU returns 0. If a number is positive, ReLU returns the number with no changes.

EXAMPLE
ReLU(-1)
# output: 0, since -1 is a negative number

ReLU(.5)
# output: .5, since .5 is not negative

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

# #### Checkpoint 1/3

# Calculate
# 
# 1. ReLU(-3)
# 2. ReLU(0)
# 3. ReLU(3)
# 
# Assign your answers as integers to `answer_1`, `answer_2`, and `answer_3` in order.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! See the `Jupyter Help` toggle at the top of the notebook for more details.

# In[1]:


## YOUR SOLUTION HERE ##
answer_1 = 0
answer_2 = 0
answer_3 = 3

# show output
print(&#39;ReLU(-3) = &#39; + str(answer_1))
print(&#39;ReLU(0) = &#39; + str(answer_2))
print(&#39;ReLU(3) = &#39; + str(answer_3))


# #### Checkpoint 2/3

# Suppose the weighted inputs to a node are
# 
# 1. -3.5
# 2. 3
# 
# Compute the output of the node in two ways:
# 
# 1. **Linear output / no activation**: compute the output of the node using no activation function. Assign your answer to the variable `Linear_output`.
# 2. **ReLU activation**: use the `ReLU` function we&#39;ve defined for you to compute the output of the node. Assign your answer to the variable `ReLU_output`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! See the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


# define the ReLU function
def ReLU(x):
    return max(0,x)

## YOUR SOLUTION HERE ##
Linear_output = -0.5
ReLU_output = 0

# show output
print(&#39;ReLU node output: &#39; + str(ReLU_output))
print(&#39;Linear node output: &#39; + str(Linear_output))


# #### Checkpoint 3/3

# Suppose the weighted inputs to a node are
# 
# 1. -2
# 2. 1
# 3. .5
# 
# We&#39;ve tried to calculate the ReLU output in the cell below, but we&#39;ve made an error. Can you fix it? The correct output is `0`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! See the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# define the ReLU function
def ReLU(x):
    return max(0,x)

## FIX OUR SOLUTION ##
ReLU_output = (-2) + (1) + (.5)

# show output
ReLU_output = ReLU(ReLU_output)


# In[ ]:




</details><script type="text/javascript" src="/relay.js"></script></body></html>
