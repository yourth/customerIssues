
# coding: utf-8

# # Customer Issue Analysis with SVM
# 
# 
# <div style="text-align: right;">copyright(c) 2016</div>
# <div style="text-align: right;">version 2.1</div>
# <div style="text-align: right;">Clinton Yourth</div>
# 

# ### ToDo:
# 
# > use genetic algorithm to optimize the penalty parameter C of the error term against the number of support vectors 

# ### Assumptions:
# 
# > using SVM with specific restrictions:
# 
# >> The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples
# 

# ### Supporting Documentation
# 
# > [wikipedia: support vector machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine)
# 
# > [wikipedia: confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
# 
# > tool docs:
# 
# >> [numpy functions](https://docs.scipy.org/doc/numpy/reference/routines.html)
# 
# >> [panda functions](http://pandas.pydata.org/pandas-docs/stable/api.html)
# 
# > scikit-learn docs:
# 
# >> Support Vector Classifier (SVC):
# 
# >>> [SVC classes](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
# 
# >>> [C-SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
# 
# >>> [linear SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
# 
# >>> [SVC examples](http://scikit-learn.org/stable/modules/svm.html#svm)
# 
# >> [metrics.classification.report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
# 
# >> [metrics.confusion.matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)
# 
# > articles:
# 
# >> [Influence of C in SVM with Linear Kernel](http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel)
# 

# In[1]:

# jupyter magic settings
get_ipython().magic(u'matplotlib inline')


# In[2]:

import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sklearn
from sklearn import preprocessing
import matplotlib as matplotlib
import matplotlib.pyplot as plt

print "python:", sys.version 
print "numpy:", np.__version__
print "pandas:", pd.__version__
print "scipy:", sp.__version__
print "scikit-learn:", sklearn.__version__
print "matplotlib:", matplotlib.__version__


# In[3]:

# import required SVM modules
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import metrics


# In[4]:

# converter for missing data
convertMissing = lambda x: float(x.strip()) if x.lstrip('-').replace('.','',1).isdigit() else np.nan


# In[5]:

# read customer issues into dataframe(df)
df = pd.read_csv('customer_issues.csv', 
                 index_col=False, 
                 header=0,
                 converters={'F1':convertMissing, 
                             'F2':convertMissing, 
                             'F3':convertMissing, 
                             'F4':convertMissing,
                             'F5':convertMissing,
                             'Target':convertMissing});
df.head()


# In[6]:

# show raw dataframe shape
df.shape


# In[7]:

# show the number of records with NaN
df[df.isnull().any(axis=1)]


# In[8]:

# drop rows with NaN
cdf = df.dropna()
cdf.head()


# In[9]:

# show cleansed dataframe shape
cdf.shape


# In[10]:

# extract target groups from dataframe
targets = cdf["Target"].values
targets[:100]


# In[11]:

# extract features from dataframe
features = cdf[["F1","F2","F3","F4","F5"]].values
features


# In[12]:

# statistical summary before scaling features
fdf = cdf[['F1','F2','F3','F4','F5']]
fdf.describe()


# In[13]:

# plot histogram of raw features
plt.hist(features, bins='auto') 
plt.show()


# In[14]:

# preprocessed data by scaling
#    - center to the mean and component wise scale to unit variance
features = preprocessing.scale(features)
features


# In[15]:

# statistical summary after scaling features
fdf = pd.DataFrame(features, columns=['F1','F2','F3','F4','F5'])
fdf.describe()


# In[16]:

# plot histogram of preprocessed features
plt.hist(features, bins='auto') 
plt.show()


# In[17]:

# plot scatter matrix for features
from pandas.tools.plotting import scatter_matrix

fdf = cdf[['F1','F2','F3','F4','F5']]
scatter_matrix(fdf, alpha=0.2, figsize=(10, 10), diagonal='kde')
print 'Features Scatter Plot'


# In[52]:

# split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

features_train, features_test, targets_train, targets_test = train_test_split(features, 
                                                                              targets, 
                                                                              test_size=0.33, 
                                                                              random_state=42)
print "features_train:\n", features_train, "\n"
print "targets_train:\n", targets_train[:100], "\n"
print "features_test:\n", features_test, "\n"
print "targets_test:\n", targets_test[:100]


# ## create SVM model

# In[53]:

# create a linear SVM classifier
clf = SVC(C=0.01, kernel='linear', decision_function_shape=None, probability=True)


# In[54]:

# get model parameters
clf.get_params(deep=True)


# In[55]:

# fit the SVM model according to the given training data
clf.fit(features_train, targets_train)


# In[56]:

# get distance of the samples features to the separating hyperplane
clf.decision_function(features_train)


# In[57]:

# support vector shape
# note: need to reduce the number of support vectors for this model
clf.support_vectors_.shape


# In[58]:

# assigned feature weights
clf.coef_


# In[59]:

# perform classification on test samples
predictions = clf.predict(features_test)
predictions[:100]


# In[60]:

# compute probabilities of possible outcomes on test samples
clf.predict_proba(features_test)[:10]


# In[62]:

# find the number of successful targets
np.sum(predictions == targets_test)


# In[63]:

# summarize the fit of the model: classification report
expected = targets_test
predicted = clf.predict(features_test)
print metrics.classification_report(expected, predicted)


# In[64]:

# summarize the fit of the model: confusion matrix
print metrics.confusion_matrix(expected, predicted)


# ## save customerIssue model

# In[65]:

from sklearn.externals import joblib

joblib.dump(clf, 'customerIssueModel.pkl', compress=9)


# ## test saved model

# In[66]:

# get saved model
savedModel = joblib.load('customerIssueModel.pkl')
savedModel


# In[67]:

# display original model
clf


# In[68]:

# check the current model feature weights
clf.coef_


# In[69]:

# check the saved model feature weights
savedModel.coef_


# In[ ]:



