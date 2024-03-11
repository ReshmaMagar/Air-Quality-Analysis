#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew


# In[2]:


df=pd.read_csv("station_day.csv")
df.head()


# In[3]:


df.shape # checking how many rows and colmns ar there


# In[4]:


df.info()


# In[10]:


df.isna().sum() # to check how many null values are in data


# In[11]:


df = df.drop(columns=['StationId','Date'])


# In[12]:


df = df[df['AQI'].isna()==False]


# In[13]:


df.isnull().sum()


# In[14]:


df.describe()


# In[ ]:


# filling all null values with mean values()
'''
df['PM2.5']=df['PM2.5'].fillna(df['PM2.5'].mean())
df['PM10']=df['PM10'].fillna(df['PM10'].mean())
df['NO']=df['NO'].fillna(df['NO'].mean())
df['NO2']=df['NO2'].fillna(df['NO2'].mean())
df['NOx']=df['NOx'].fillna(df['NOx'].mean())
df['NH3']=df['NH3'].fillna(df['NH3'].mean())
df['CO']=df['CO'].fillna(df['CO'].mean())
df['SO2']=df['SO2'].fillna(df['SO2'].mean())
df['O3']=df['O3'].fillna(df['O3'].mean())
df['Benzene']=df['Benzene'].fillna(df['Benzene'].mean())
df['Toluene']=df['Toluene'].fillna(df['Toluene'].mean())
df['Xylene']=df['Xylene'].fillna(df['Xylene'].mean())
df['AQI']=df['AQI'].fillna(df['AQI'].mode()[0])
df['AQI_Bucket']=df['AQI_Bucket'].fillna('Moderate')
'''


# In[15]:


#step 1: replace "BLANK" with NAN
df['PM2.5'].replace(' ',np.nan,inplace=True)
df['PM10'].replace(' ',np.nan,inplace=True)
df['NO'].replace(' ',np.nan,inplace=True)
df['NO2'].replace(' ',np.nan,inplace=True)
df['NOx'].replace(' ',np.nan,inplace=True)
df['NH3'].replace(' ',np.nan,inplace=True)
df['CO'].replace(' ',np.nan,inplace=True)
df['SO2'].replace(' ',np.nan,inplace=True)
df['O3'].replace(' ',np.nan,inplace=True)
df['Benzene'].replace(' ',np.nan,inplace=True)
df['Toluene'].replace(' ',np.nan,inplace=True)
df['Xylene'].replace(' ',np.nan,inplace=True)
df['AQI'].replace(' ',np.nan,inplace=True)
df['AQI_Bucket'].replace(' ','Moderate',inplace=True)


# In[16]:


df.info()


# In[17]:


# Calculate mean values

pm2mean=df['PM2.5'].mean()
pm1mean=df['PM10'].mean()
nomean=df['NO'].mean()
noomean=df['NO2'].mean()
noxmean=df['NOx'].mean()
nhmean=df['NH3'].mean()
comean=df['CO'].mean()
somean=df['SO2'].mean()
omean=df['O3'].mean()
benmean=df['Benzene'].mean()
xymean=df['Xylene'].mean()
tolmean=df['Toluene'].mean()
aqimean=df['AQI'].mode()[0]


# In[18]:


df.info()


# In[ ]:


df.head(25)


# In[ ]:


#df=df.drop(['AQI_Bucket'],axis=1,inplace=True)


# In[ ]:


df.info()


# In[20]:


df=df.drop(['AQI_Bucket'],axis=1,inplace=True)


# In[21]:


# selecting feature and target

X=df.iloc[:,:-1]  # feature---->selecting all the colmns except last colmn(feature)

y=df.iloc[:,-1] # target---->selecting all teh rows of last column(response)


# In[22]:


X=df.drop(['AQI'],axis=1)
y=df['AQI']


# In[ ]:




