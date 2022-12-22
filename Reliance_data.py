#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('reliance_data.csv')


# In[3]:


df


# In[4]:


df.info()


# Columns: Last, Trades, Deliverable Volume, %Deliverables have missing values 

# In[5]:


df.describe()


# # converting the date column to datetime

# In[6]:


df['Date'] = pd.to_datetime(df['Date'])


# In[7]:


df.info()


# # Checking for unique values in symbol and series columns 

# In[8]:


print('Total Unique values in Symbol are: ', df['Symbol'].nunique())


# In[9]:


print("Total Unique values  in Series are: ", df['Series'].nunique())


# # Since both the columns have zero variance, we drop them

# In[10]:


df = df.drop(['Symbol','Series'], axis = 1)


# # Missing values analysis 

# In[11]:


missing_values  = df.isna().sum()


# In[12]:


missing_values


# # In this Scenario, we have to in-depth analysis of each column displaying missing values, to understand what kind of treatment is required

# In[13]:


sns.boxplot(df['Last'])


# # 'Last' column has outliers, in this context mean imputation is not applicable, so we can go with median imputation techniques 

# In[14]:


from sklearn.impute import SimpleImputer

SI = SimpleImputer(missing_values=np.nan, strategy='median')

df['Last'] = SI.fit_transform(df[['Last']])


# # Now doing analysis for Trades column

# In[15]:


sns.boxplot(df['Trades'])


# # This column also has outliers 

# In[16]:


df['Trades'] = SI.fit_transform(df[['Trades']])


# In[17]:


df.isna().sum()


# # Now doing analysis for Deliverable Volume and %Deliverable, to see if its required to keep both the columns

# In[18]:


corr = df.corr()


# In[19]:


corr


# # Deliverable Volume and %Deliverable have poor correlation, in this context, we can do imputation for both

# In[20]:


sns.boxplot(df['Deliverable Volume'])


# In[21]:


sns.boxplot(df['%Deliverble'])


# # Althought %Deliverable does not have a lot of outliers, we can stick with using median imputation

# In[22]:


df[['Deliverable Volume', '%Deliverble']] = SI.fit_transform(df[['Deliverable Volume','%Deliverble']])


# In[23]:


df.isna().sum()


# # Now that outliers are done, we will shift the focus towards outlier analysis

# In[24]:


for i in df:
    
    if (df[i].dtypes == np.float64) or (df[i].dtypes == np.int64):
        
        plt.figure()
        sns.boxplot(df[i])


# In[25]:


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()


# In[26]:


df


# In[27]:


df.iloc[:, 1:] = sc.fit_transform(df.iloc[:, 1:])


# In[28]:


df = df.drop(['Date'],axis =1)


# In[29]:


train = df.iloc[0:5974, :]
test = df.iloc[5974:6205, :]


# In[30]:


train


# In[31]:


X_train = train.drop(['Close'], axis=1)
Y_train = train[['Close']]


# In[32]:


X_test = test.drop(['Close'], axis =1)
Y_test = test[['Close']]


# In[33]:


X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape


# # Building LSTM model

# In[34]:


import tensorflow as tf 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[39]:


model = Sequential()
model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dense(25, activation = 'sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# In[40]:


model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[41]:


history = model.fit(X_train, Y_train, batch_size = 64, epochs = 10, validation_data = (X_test, Y_test))


# In[44]:


plt.plot(history.history['loss'], label ='train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()


# In[45]:


y_pred = model.predict(X_test)


# In[46]:


y_pred


# In[49]:


X_test.shape[0]


# In[52]:


X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
data = np.concatenate((X_test, y_pred), axis = 1)
inv_data = sc.inverse_transform(data)
y_pred = inv_data[:,-1]


# In[55]:


plt.figure()
plt.plot(df.Close)
index = list(range(5974,6205))
plt.plot(index, y_pred, label = 'LSTM predicted')
plt.legend()


# In[ ]:




