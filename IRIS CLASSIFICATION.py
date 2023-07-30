#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris = pd.read_csv("IRIS.csv")


# In[6]:


iris.head()


# In[7]:


iris.describe()


# In[9]:


iris.shape


# In[10]:


iris.groupby('species').mean()


# In[11]:


sns.scatterplot(x='sepal ength', y='sepal width', hue='species', data=iris)
plt.show()


# In[12]:


sns.lineplot(data=iris.drop(['species'], axis=1))
plt.show()


# In[13]:


iris.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
plt.show()


# In[14]:


g = sns.FacetGrid(iris, col='species')
g = g.map(sns.kdeplot, 'sepal_length')


# In[15]:


iris.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
x = iris.drop('species', axis=1)
y= iris.species
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# In[17]:


sns.pairplot(iris)


# In[20]:


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

svm.score(x_test, y_test)


# In[ ]:




