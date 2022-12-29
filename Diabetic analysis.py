#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("C:/Users/Sathvik K/OneDrive/Desktop/diabetes.csv")
df


# In[4]:


df.head()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[12]:


df['Outcome'].value_counts()


# In[13]:


df.groupby('Outcome').mean()


# In[16]:


#droping a row-->axis =0
#droping a column -->axis =1
X = df.drop(columns = 'Outcome',axis = 1)
Y = df['Outcome']
X


# In[17]:


Y


# In[18]:


#Data Standardization
scaler = StandardScaler()


# In[19]:


scaler.fit(X)


# In[21]:


standardisedData = scaler.transform(X)
standardisedData


# In[22]:


print(standardisedData)


# In[26]:


X = standardisedData
Y = df['Outcome']

print(X,Y)


# In[27]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, stratify = Y,random_state = 2)


# In[29]:


print(X_train)
print(X_test)


# In[30]:


classifier = svm.SVC(kernel = 'linear')


# In[31]:


classifier.fit(X_train,Y_train)


# In[46]:


#accuracy score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
training_data_accuracy


# In[47]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
test_data_accuracy


# In[62]:


input_data = (4,1000,92,0,0,37.6,0.191,30)
#change to numpy array

input_data_in_numpy = np.asarray(input_data)
input_data_in_numpy


# In[63]:


#reshape the array

input_data_reshaped = input_data_in_numpy.reshape(1,-1)


# In[64]:


#standaredising the input single data

std_data = scaler.transform(input_data_reshaped)
print(std_data)


# In[65]:


prediction = classifier.predict(std_data)
# print(prediction)


if(prediction[0] == 0):
    print('person is not diabetic')
else:
    print('person is diabetic')

