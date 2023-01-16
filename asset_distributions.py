#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Plots, formats and normalizes the daily / monthly returns of a given index or stock and compares with FED FUNDs Rate (%)

import time, datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange
import numpy as np

parent_dir = Path.cwd().parent.absolute()

csv_dir = parent_dir / 'csv'

todays_date = datetime.date.today()
ticker = '^GSPC'
#ticker = 'FMG.AX'
csv_file = ticker + '-yahoo-finance-' + str(todays_date) + '.csv'
csv_file_absolute = csv_dir / csv_file

print('\nparent_dir', parent_dir)
print('csv_dir', csv_dir)
print(f'csv_dir: {csv_dir}')

print('todays_date', todays_date)
print('ticker:', ticker)
print('csv_file', csv_file)
print('csv_file_absolute', csv_file_absolute, '\n')

def normalize_original(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

YEAR = 1986
MONTH = 1
DAY = 1

period1 = int(time.mktime(datetime.datetime(YEAR, MONTH, DAY, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.date.today().timetuple()))

interval = '1d' # 1wk, 1m

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

# treat query string as an api call
print('\nperiod1:', period1)
print('period2:', period2)
print('query_string:', query_string, '\n')

df = pd.read_csv(query_string)
df3 =  pd.read_csv("FEDFUNDS.csv")


# In[3]:


df = df[['Date','Adj Close']]


# In[30]:


df.rename(columns={'Date' : 'date', 'Adj Close' : 'price_t'}, inplace = True)


# In[31]:


df['% returns'] = df['price_t'].pct_change(1)


# In[32]:


df.set_index('date',inplace = True)


# In[33]:


df['% returns'].plot(figsize=(12,8))


# In[34]:


# df = df.reset_index(drop=True)
df = df.reset_index()
print(df.head())


# In[35]:


print(f'len(df): {len(df)}')
print(f'len(df3): {len(df3)}')

months = len(df) / 12
print(months)


# In[36]:


print(df.head())


# In[37]:


df['date'] =  pd.to_datetime(df['date'], format='%Y-%m-%d')
df3.head()


# In[38]:



df3['date'] =  pd.to_datetime(df3['date'], format='%d/%m/%Y')


# In[39]:


result = pd.merge(df, df3, how="outer", on=["date"])
print(result.head())


# In[40]:


result['date'] =  pd.to_datetime(result['date'], format='%Y-%m-%d')
result = result.sort_values(by=['date'])
result['price_t'] = result['price_t'].interpolate(method='linear')
result['% returns'] = result['% returns'].interpolate(method='linear')


# In[41]:


#result.to_csv('result_c.csv', index=False)
greater_than_3 = result['fed_funds (%)'] > 3
df9 = result[greater_than_3]


# In[42]:


def normalize(df, feature_name):
    result = df.copy()
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[43]:


result = normalize(df=result, feature_name='% returns')
result = normalize(df=result, feature_name='price_t')
result = normalize(df=result, feature_name='fed_funds (%)')
print(result)


# In[44]:


result['fed_funds (%)'] = result['fed_funds (%)'].fillna(method="ffill")
result.plot(x = 'date', y = ['price_t', 'fed_funds (%)'],figsize = (20,20))


# In[45]:


result.plot(x = 'date', y = ['% returns', 'fed_funds (%)'],figsize = (20,20))


# In[46]:


print(f'df.shape: {df.shape}')
print(f'df3.shape: {df3.shape}')
print(f'result.shape: {result.shape}')


# In[47]:


#Above calculates the % returns of any stock on yahoo. 
#Below Calculates Average historic returns. 
df['price_t'].plot(figsize=(12,8))
df3['fed_funds (%)']


# In[48]:


fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(1, 1, 1)
df['% returns'].hist(bins=40, ax=ax1)
ax1.set_xticks(np.arange(-0.2,0.1,0.01))
ax1.set_yticks(np.arange(0,1000,100))
ax1.set_xlabel('Return')
ax1.set_ylabel('Samples')
#ax1.set_title('Return distribution'")
plt.show()


# In[49]:


#Stats on % Daily Returns of Selected Ticker. 
df['% returns'].describe()


# In[50]:


#Declare new DF for Annualized Expected Return (Average) 
df2 = df['% returns'].describe()
print("The Daily Expected Return is", df2.loc['mean'], "%")
daily = df2.loc['mean']


# In[51]:


#Anualized Daily Expected Return Calculation is. 
Anualized_R_Simple = daily*250
Anualized_R_Compounded = ((1+daily)**250)-1

print("Anualized_R_Simple = ",Anualized_R_Simple*100,"%")
print("Anualized_R_Compounded",Anualized_R_Compounded*100, "%")


# In[56]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
df9['% returns'].hist(bins=40, ax=ax1)
ax1.set_xlabel('Return')
ax1.set_yticks(range(0,30))

ax1.set_ylabel('Samples')
ax1.set_title('Return distribution')
plt.show()


# In[57]:


df9['% returns'].describe()


# In[ ]:




