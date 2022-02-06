#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# In[11]:


dataset = pd.read_csv(r"C:/Users/tanvi/Downloads/Google_Stock_Price_Train.csv",index_col="Date",parse_dates=True)


# In[12]:


dataset.head()


# In[13]:


dataset.isna().any()


# In[14]:


dataset.info()


# In[15]:


dataset['Open'].plot()


# In[16]:


dataset['Close']=dataset['Close'].str.replace(',', '').astype(float)


# In[17]:


dataset['Volume']=dataset['Volume'].str.replace(',', '').astype(float)


# In[18]:


dataset.info()


# In[19]:


dataset.rolling(7).mean().head(20)


# In[20]:


dataset['Open'].plot()
dataset.rolling(window=30).mean()['Close'].plot()


# In[21]:


dataset['Close : 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close' , 'Close : 30 Day Mean']].plot()


# In[22]:


training_set = dataset['Open']
training_set = pd.DataFrame(training_set)


# In[23]:


dataset.isna().any()


# In[24]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[25]:


x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i , 0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[27]:


regressor = Sequential()


# In[28]:


#add first LSTM layer and dropout regularisation, providing input
regressor.add(LSTM(units = 50, return_sequences = True, input_shape =(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#add second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#add third LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#add fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#add output layer
regressor.add(Dense(units=1))


# In[29]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(x_train, y_train, epochs = 100, batch_size =32)


# In[31]:


dataset_test = pd.read_csv(r"C:/Users/tanvi/Downloads/Google_Stock_Price_Test.csv",index_col="Date",parse_dates=True)


# In[32]:


real_stock_price = dataset_test.iloc[:, 1:2].values


# In[33]:


dataset_test.head()


# In[34]:


dataset_test.info()


# In[35]:


dataset_test['Volume']=dataset_test['Volume'].str.replace(',', '').astype(float)


# In[36]:


test_set = dataset_test['Open']
test_set = pd.DataFrame(test_set)


# In[37]:


test_set.info()


# In[46]:


#prediction
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[47]:


predicted_stock_price = pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()


# In[49]:


#Visualizing
plt.plot(real_stock_price, color = 'red' , label = 'Real Googlr Stock Price')
plt.plot(predicted_stock_price, color = 'blue' , label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show() 


# In[ ]:




