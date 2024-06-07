<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# ## Wrapper Methods
# 
# In this project, you&#39;ll analyze data from a survey conducted by Fabio Mendoza Palechor and Alexis de la Hoz Manotas that asked people about their eating habits and weight. The data was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+). Categorical variables were changed to numerical ones in order to facilitate analysis.
# 
# First, you&#39;ll fit a logistic regression model to try to predict whether survey respondents are obese based on their answers to questions in the survey. After that, you&#39;ll use three different wrapper methods to choose a smaller feature subset.
# 
# You&#39;ll use sequential forward selection, sequential backward floating selection, and recursive feature elimination. After implementing each wrapper method, you&#39;ll evaluate the model accuracy on the resulting smaller feature subsets and compare that with the model accuracy using all available features.

# In[50]:


# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
get_ipython().run_line_magic(&#39;matplotlib&#39;, &#39;inline&#39;)


# ## Evaluating a Logistic Regression Model
# 
# The data set `obesity` contains 18 predictor variables. Here&#39;s a brief description of them.
# 
# * `Gender` is `1` if a respondent is male and `0` if a respondent is female.
# * `Age` is a respondent&#39;s age in years.
# * `family_history_with_overweight` is `1` if a respondent has family member who is or was overweight, `0` if not.
# * `FAVC` is `1` if a respondent eats high caloric food frequently, `0` if not.
# * `FCVC` is `1` if a respondent usually eats vegetables in their meals, `0` if not.
# * `NCP` represents how many main meals a respondent has daily (`0` for 1-2 meals, `1` for 3 meals, and `2` for more than 3 meals).
# * `CAEC` represents how much food a respondent eats between meals on a scale of `0` to `3`.
# * `SMOKE` is `1` if a respondent smokes, `0` if not.
# * `CH2O` represents how much water a respondent drinks on a scale of `0` to `2`.
# * `SCC` is `1` if a respondent monitors their caloric intake, `0` if not.
# * `FAF` represents how much physical activity a respondent does on a scale of `0` to `3`.
# * `TUE` represents how much time a respondent spends looking at devices with screens on a scale of `0` to `2`.
# * `CALC` represents how often a respondent drinks alcohol on a scale of `0` to `3`.
# * `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, and `Walking` indicate a respondent&#39;s primary mode of transportation. Their primary mode of transportation is indicated by a `1` and the other columns will contain a `0`.
# 
# The outcome variable, `NObeyesdad`, is a `1` if a patient is obese and a `0` if not.
# 
# Use the `.head()` method and inspect the data.

# In[51]:


# https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

# Load the data
obesity = pd.read_csv(&#34;obesity.csv&#34;)

# Inspect the data
obesity.head()


# ### Split the data into `X` and `y`
# 
# In order to use a linear regression model, you&#39;ll need to split the data into two parts: the predictor variables and an outcome variable. Do this by splitting the data into a DataFrame of predictor variables called `X` and a Series of outcome variables `y`.

# In[52]:


X = obesity.drop([&#34;NObeyesdad&#34;], axis=1)
y = obesity[&#34;NObeyesdad&#34;]


# ### Logistic regression model
# 
# Create a logistic regression model called `lr`. Include the parameter `max_iter=1000` to make sure that the model will converge when you try to fit it.

# In[53]:


lr = LogisticRegression(max_iter = 1000)


# ### Fit the model
# 
# Use the `.fit()` method on `lr` to fit the model to `X` and `y`.

# In[54]:


lr.fit(X,y)


# ### Model accuracy
# 
# A model&#39;s _accuracy_ is the proportion of classes that the model correctly predicts. is Compute and print the accuracy of `lr` by using the `.score()` method. What percentage of respondents did the model correctly predict as being either obese or not obese? You may want to write this number down somewhere so that you can refer to it during future tasks.

# In[55]:


lr.score(X, y)  # the model correctly predicts the class of about 76% of the respondents


# ## Sequential Forward Selection
# 
# Now that you&#39;ve created a logistic regression model and evaluated its performance, you&#39;re ready to do some feature selection. 
# 
# Create a sequential forward selection model called `sfs`. 
# * Be sure to set the `estimator` parameter to `lr` and set the `forward` and `floating` parameters to the appropriate values. 
# * Also use the parameters `k_features=9`, `scoring=&#39;accuracy&#39;`, and `cv=0`.

# In[56]:


sfs = SFS(lr, 
          k_features=9, 
          forward=True, 
          floating=False, 
          scoring=&#39;accuracy&#39;,
          cv=0)
# choosing the best 9 features, forward selection,
# scoring based on the accuracy of the model
# no cross validation should be used


# ### Fit the model
# 
# Use the `.fit()` method on `sfs` to fit the model to `X` and `y`. This step will take some time (not more than a minute) to run.

# In[57]:


sfs.fit(X,y)


# ### Inspect the results
# 
# Now that you&#39;ve run the sequential forward selection algorithm on the logistic regression model with `X` and `y` you can see what features were chosen and check the model accuracy on the smaller feature set. Print `sfs.subsets_[9]` to inspect the results of sequential forward selection.

# In[58]:


print(sfs.subsets_[9])


# ### Chosen features and model accuracy
# 
# Use the dictionary `sfs.subsets_[9]` to print a tuple of chosen feature names. Then use it to print the accuracy of the model after doing sequential forward selection. How does this compare to the model&#39;s accuracy on all available features?

# In[59]:


print(sfs.subsets_[9][&#39;feature_names&#39;])
print(sfs.subsets_[9][&#39;avg_score&#39;])


# ### Visualize model accuracy
# 
# It can be helpful to visualize the results of sequential forward selection and see how accuracy is affected as each feature is added. Use the code `plot_sfs(sfs.get_metric_dict())` to plot the model accuracy as a function of the number of features used. Make sure to show your plot as well.

# In[60]:


plot_sfs(sfs.get_metric_dict())
plt.show()


# ## Sequential Backward Selection

# Sequential forward selection was able to find a feature subset that performed marginally better than the full feature set. Let&#39;s use a different sequential method and see how it compares.
# 
# Create a sequential backward selection model called `sbs`. 
# * Be sure to set the `estimator` parameter to `lr` and set the `forward` and `floating` parameters to the appropriate values.
# * Also use the parameters `k_features=7`, `scoring=&#39;accuracy&#39;`, and `cv=0`.

# In[78]:


sbs = SFS(lr, 
          k_features=7, 
          forward=False, 
          floating=False, 
          scoring=&#39;accuracy&#39;,
          cv=0)
# choosing the best 7 features, backward selection,
# scoring based on the accuracy of the model
# no cross validation should be used


# ### Fit the model
# 
# Use the `.fit()` method on `sbs` to fit the model to `X` and `y`.

# In[79]:


sbs.fit(X, y)


# ### Inspect the results
# 
# Now that you&#39;ve run the sequential backward selection algorithm on the logistic regression model with `X` and `y` you can see what features were chosen and check the model accuracy on the smaller feature set. Print `sbs.subsets_[7]` to inspect the results of sequential backward selection.

# In[63]:


print(sbs.subsets_[7])


# ### Chosen features and model accuracy
# 
# Use the dictionary `sbs.subsets_[7]` to print a tuple of chosen feature names. Then use it to print the accuracy of the model after doing sequential backward selection. How does this compare to the model&#39;s accuracy on all available features?

# In[64]:


print(sbs.subsets_[7][&#39;feature_names&#39;])
print(sbs.subsets_[7][&#39;avg_score&#39;])


# ### Visualize model accuracy
# 
# You can visualize the results of sequential backward floating selection just as you did with sequential forward selection. Use the code `plot_sfs(sbs.get_metric_dict())` to plot the model accuracy as a function of the number of features used.

# In[66]:


plt.clf()
plot_sfs(sbs.get_metric_dict())
plt.show()


# ## Recursive Feature Elimination

# So far you&#39;ve tried two different sequential feature selection methods. Let&#39;s try one more: recursive feature elimination. First you&#39;ll standardize the data, then you&#39;ll fit the RFE model and inspect the results.
# 
# At a later step of this project, you&#39;ll need to be able to access feature names. Enter the code `features = X.columns` for use later.

# In[67]:


features = X.columns


# ### Standardize the data
# 
# Before doing applying recursive feature elimination it is necessary to standardize the data. Standardize `X` and save it as a DataFrame by creating a `StandardScaler()` object and using the `.fit_transform()` method.

# In[69]:


X = pd.DataFrame(StandardScaler().fit_transform(X))
# perform scaling


# ### Recursive feature elimination model
# 
# Create an `RFE()` object that selects `8` features. Be sure to set the `estimator` parameter to `lr`.

# In[71]:


rfe = RFE(estimator=lr, n_features_to_select=8)


# ### Fit the model
# 
# Fit the recursive feature elimination model to `X` and `y`.

# In[72]:


rfe.fit(X,y)


# ### Inspect chosen features
# 
# Now that you&#39;ve fit the RFE model you can evaluate the results. Create a list of chosen feature names and call it `rfe_features`. You can use a list comprehension and filter the features in `zip(features, rfe.support_)` based on whether their support is `True` (meaning the model kept them) or `False` (meaning the model eliminated them).

# In[76]:


rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]
print(rfe_features)


# ### Model accuracy
# 
# Use the `.score()` method on `rfe` and print the model accuracy after doing recursive feature elimination. How does this compare to the model&#39;s accuracy on all available features?

# In[74]:


rfe.score(X,y)


# In[ ]:




<script type="text/javascript" src="/relay.js"></script></body></html>
