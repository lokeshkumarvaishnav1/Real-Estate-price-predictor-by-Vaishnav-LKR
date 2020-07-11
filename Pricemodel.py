#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


hdata=pd.read_csv("Real_estate.csv")


# In[3]:


hdata.columns=['No','transaction date','house age','MRT station','number of convenience stores','latitude','longitude','house price of unit area']


# In[4]:


hdata.head().transpose()


# In[5]:


hdata.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
hdata.hist(bins=15,figsize=(20,15))


# In[7]:


corrmatrix=hdata.corr()
corrmatrix['house price of unit area'].sort_values(ascending=False)


# In[8]:


import seaborn as sns
sns.heatmap(corrmatrix,annot=True)


# In[9]:


from pandas.plotting import scatter_matrix
attributes = ['No','transaction date','house age','MRT station','number of convenience stores','latitude','longitude','house price of unit area']
scatter_matrix(hdata[attributes], figsize = (20,15))


# In[10]:


hdata.info()


# In[11]:


hdata.isna()


# In[12]:


#housing = hdata.drop("house price of unit area", axis=1)
#housing_labels = np.array(hdata["house price of unit area"].copy()).reshape(414,1)
#print(housing.shape, housing_labels.shape)


# In[13]:


hdata_X=hdata.iloc[:,:-1]
hdata_Y=hdata.iloc[:,-1]
print(hdata_X.shape ,hdata_Y.shape)


# In[14]:


hdata_X.head()


# In[15]:


hdata_Y#.head()


# In[16]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(hdata_X,hdata_Y,test_size=0.15,random_state=42)
#print("Rows in train set:",len(train_set), "\nRows in test set:",len(test_set))


# In[17]:


"""
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
sss.get_n_splits(hdata_X, hdata_Y)
StratifiedShuffleSplit(n_splits=5, random_state=42)
for train_index, test_index in sss.split(hdata_X, hdata_Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]"""


# In[18]:


from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
    ('normalizer',Normalizer()),
])


# In[19]:


hdata_X= my_pipeline.fit_transform(xtrain)


# In[20]:


hdata_X.shape


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(xtrain,ytrain)


# In[22]:


y_pred=model.predict(xtest)
y_pred


# In[23]:


import numpy as np
a=np.array(ytest)
a


# In[24]:


#from sklearn.metrics import accuracy_score
#score=accuracy_score(ytest,y_pred)
#y_pred=y_pred.reshape(63,1)


# In[25]:


#model.score(y_pred,ytest)


# In[26]:


model.score(xtrain,ytrain)


# In[27]:


model.score(xtest,ytest)


# # saving model in pickle

# In[28]:


import pickle
pickle_out = open("Pricemodel.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# # using model

# In[33]:


testdata=pd.read_csv('testdata.csv')


# In[44]:


pickle_in = open("Pricemodel.pkl","rb")
model=pickle.load(pickle_in)


# In[45]:


model.predict(testdata)


# 
