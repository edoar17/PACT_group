#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:06:39 2022

@author: edoar17
"""

import pandas as pd
import statsmodels.api as sm

def log_return(series):
    return np.log(series).diff()
# Inflation, AUD, ASX200 on PGH

#Get Data from  Folder
freq = '_quarterly'
asx = pd.read_csv('data/ASX'+freq).dropna()
asx['Date'] = pd.to_datetime(asx['Date'])
asx = asx.set_index('Date')
asx['year'] = asx.index.year
asx['quarter'] = asx.index.quarter
asx = asx.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1)

aud = pd.read_csv('data/AUD_strength'+freq)
aud['Date'] = pd.to_datetime(aud['Date'])
aud = aud.set_index('Date')
aud['year'] = aud.index.year
aud['quarter'] = aud.index.quarter
aud['return'] = log_return(aud['cumret'])
aud = aud.dropna()

pgh = pd.read_csv('data/PGH.AX'+freq)
pgh['Date'] = pd.to_datetime(pgh['Date'])
pgh = pgh.set_index('Date')
pgh['year'] = pgh.index.year
pgh['quarter'] = pgh.index.quarter
pgh = pgh.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1)


inflation = pd.read_csv('data/inflation.csv')
inflation = inflation.rename(columns = {'Year': 'year'})

#Regression table
pgh = pgh.set_index(['year', 'quarter'])
aud = aud.set_index(['year', 'quarter'])
asx = asx.set_index(['year', 'quarter'])
inflation = inflation.set_index(['year', 'quarter'])

table = pgh.join(aud['return'], how='inner', rsuffix='_aud').join(
    asx['return'], rsuffix='_asx', how='inner').join(
    inflation['pct_change'])

table.corr()
# remove inflation, too correlated with others
table = table.drop('pct_change', axis=1)
table.corr()        
        
# Do regression
y = table['return']
X = table.drop('return', axis=1)
X = sm.add_constant(X)
lm = sm.OLS(y, X).fit()      
lm.summary()
results = lm.summary().as_text()
with open('regression_results.txt', mode='w') as f:
    f.write(results)

### AUD fits well
### Regression PGH vs ASX
y = table['return']
X = table['return_asx']
X = sm.add_constant(X)
lm = sm.OLS(y, X).fit()      
lm.summary()














