#!/usr/bin/env python
# coding: utf-8

# The Oasis Infobyte: Data Science Internship
# 
# Task 1 : iris flower classification with Machine Learning.
# 
# Author: Dhiraj Shailesh Phalke

# In[ ]:





# Steps to Classify Iris Flower:
# 
# 1 Load the data.
# 
# 2 Clean the data.
# 
# 3 Analyze and visualize the dataset.
# 
# 4 Model training.
# 
# 5 Model Evaluation.
# 
# 

# In[ ]:





# Step 1 : Load the data

# In[1]:


# Iris Flower Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data= pd.read_csv("C://Users/dhira/OneDrive/Desktop/Iris.csv")
data


# First, we’ve imported some necessary packages for the project.
# 
# Numpy will be used for any computational operations.
# 
# We’ll use Matplotlib and seaborn for data visualization.
# 
# Pandas help to load data from various sources like local storage, database, excel file, CSV file, etc.
# 
# Next, we load the data using pd.read_csv() and set the column name as per the iris data information.
# 
# Pd.read_csv reads CSV files. CSV stands for comma separated value.
# 
# df.head() only shows the first 5 rows from the data set table.

# In[4]:


data.head()


# Step 2 : Clean the data

# In[5]:


#Dropping unnecessary column
mydata=data.drop(columns=['Id'])
mydata


# In[ ]:





# Step 3 – Analyze and visualize the dataset:

# In[6]:


# Some basic statistical analysis about the data
mydata.describe()


# From this description, we can see all the descriptions about the data, like average length and width,minimum value, maximum value, the 25%, 50%, and 75% distribution value, etc.

# In[7]:


sns.pairplot(mydata)


# In[8]:


# Separate features and target  
df = mydata.values
X = df[:,0:4]
Y = df[:,4]


# In[10]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 


# Here we separated the features from the target value.

# In[11]:


# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# In[12]:


# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# Here we can clearly see the verginica is the longest and setosa is the shortest flower.

# Step 4 – Model training:

# In[13]:


#importing libararies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings


# In[14]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# Using train_test_split we split the whole data into training and testing datasets. Later we’ll use the testing dataset to check the accuracy of the model.

# In[15]:


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# Step 5 – Model Evaluation:

# In[16]:


# Predict from the test dataset
predictions = svn.predict(X_test)
# Calculate the accuracy
accuracy_score(y_test, predictions)


# Output: 1
# 
# The accuracy is 100%.

# In[17]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# The classification report gives a detailed report of the prediction. Precision defines the ratio of true positives to the sum of true positive and false positives. Recall defines the ratio of true positive to the sum of true positive and false negative. F1-score is the mean of precision and recall value. Support is the number of actual occurrences of the class in the specified dataset.

# In[ ]:




