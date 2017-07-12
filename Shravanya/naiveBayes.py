
# coding: utf-8

# In[17]:

#implementing classification algorithm Naive Bayes using sklearn package
#classifying a given protein molecule into the number of protein chains it might have
#this is a multiple classification problem as the model is trained on many attributes

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

#read the csv files for both training and testing
ftrain = pd.read_csv('C:/Users/user/Documents/hitachi pentaho/train.csv')
ltrain = pd.read_csv('C:/Users/user/Documents/hitachi pentaho/trainVar.csv')
ftest = pd.read_csv('C:/Users/user/Documents/hitachi pentaho/test.csv')
ltest = pd.read_csv('C:/Users/user/Documents/hitachi pentaho/testVar.csv')

#creating data frames for training purposes
#x will take all the features and y will take the labels
x = ftrain.values.tolist()
y = list(ltrain['molecules'])

#creating data frames for testing purposes
#p will takeall the features and q will be the expected classification
p = ftest.values.tolist()
q = list(ltest['molecules'])

clf = GaussianNB()
clf.fit(x, y)
output = clf.predict(p)
print(output)
print(accuracy_score(q, output))


# In[ ]:



