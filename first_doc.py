#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 01:06:34 2022

@author: edoar17
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import dataframe_image as dfi
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

def log_return(series):
    return np.log(series).diff()

ticker = 'PGH.AX'

PGH_ = yf.Ticker(ticker)
pgh = PGH_.history(period='3y', interva='1d')

def calcs(df):
    df['log_return'] = log_return(df['Close'])
    df['cum_ret'] = np.exp(df['log_return'].cumsum())
    df = df.dropna()
    return df

# Graph price action
def graph1(df, title):
    x = df.index
    y = np.array(df.Close)
    y1 = np.array(df.cum_ret-1)
    
    fig, axs = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios': [4,1]})
    
    axs[0].plot(x,y)
    axs[0].set_title(title)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.15, 0.9, f'Cumulative Return is: {df.cum_ret[-1].round(4)}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[0].transAxes, bbox=props)
    colors = ['g' if i>=0 else 'r' for i in y1]
    axs[1].bar(x, y1, color=colors)
    axs[1].set_title('Cumulative Return')
    
    path = f"plots/{title}.png"
    if not os.path.exists(path):
        fig.savefig(f"plots/{title}.png", format='png')
    fig.show()


pgh = calcs(pgh)
graph1(pgh, 'Price of PGH.AX')

# From pandemic low
pandemic_bottom = '2020-03-20'
pgh1 = calcs(pgh.loc[pgh.index>=pandemic_bottom])
graph1(pgh1, 'Price of PGH.AX from pandemic low')

# Check relevant ETFs
indexes = {'EWA': 'iShares MSCI-Australia ETF',
           '^AXJO': 'ASX 200',
           'FAIR.AX': 'Betashares Australian Sustainability Leaders ETF ',
           'MVS.AX': 'VanEck Vectors Small Companies Masters ETF',
           'DBC': 'Invesco DB Commodity Index Tracking Fund',
           'DBO': 'Invesco DB Oil Fund'}

currencies = {'AUDUSD=X': 'AUD/USD',
              'AUDJPY=X': 'AUD/JPY'}

                # PPG.AX 800M
competitors = {'PPG.AX': 'Pro-Pac Packaging Limited', #101M
               'ORA.AX': 'Orora Limited', #3B
               'AMC.AX': 'Amcor plc'} #23B


#Linear regressions
def prep_features(df, df2):
    # df = df.loc[df.index>=pandemic_bottom]

    data = df[['log_return']].join(df2['log_return'], rsuffix='_pact')
    data = data.dropna()
    data['log_return'] = data['log_return']/data['log_return'].max()
    data['log_return_pact'] = data['log_return_pact']/data['log_return_pact'].max()
    return data

# plot returns
def graph2(df, title):
    #Regression
    x = df.iloc[:,0] #X's log_returns
    y = df.iloc[:,1] #Pact is the y

    lm = sm.OLS(y, x)
    reg = lm.fit()    
    pred = reg.predict(x)
    summary_text = reg.summary().as_text()
    
    #Plotting
    x = np.array(df.iloc[:,0]).reshape(-1, 1) #X's log_returns
    y = np.array(y).reshape(-1, 1)
    pred = np.array(pred).reshape(-1, 1) #Pact is the y

    rango = [y.min(), y.max()]

    fig, axs = plt.subplots(2,1, figsize=(12,12))
    axs[0].grid(alpha=0.5, linestyle='dashed', zorder=-1)
    axs[0].scatter(x,y)
    axs[0].plot(x, pred, color='r')
    axs[0].set_xlim(rango)
    axs[0].set_ylim(rango)
    axs[0].set_title(title)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[1].set_axis_off()
    axs[1].text(0.5, 0.5, summary_text, 
             horizontalalignment='center', verticalalignment='center', 
             bbox=props, size=10)
    fig.tight_layout()

    path = f"plots/{title}.png"
    # if not os.path.exists(path):
    fig.savefig(path, format='png')
    fig.show()



# Add performance
performance = pd.DataFrame(columns=['over', 'value'])
prices = pgh1[['log_return']]
cum_ret = pgh1[['cum_ret']]

for i in list(indexes):
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interval='1d')
    #Calculate returns, cum_ret
    bench = calcs(bench.loc[bench.index>=pandemic_bottom])
    
    #Save data to two dfs
    col_name = i+'_log_return'
    col_name1 = i+'_cum_ret'
    prices[col_name] = bench.log_return
    cum_ret[col_name1] = bench.cum_ret
    
    #Add overperformance of PGH relative
    row = {'over': i, 
           'value': pgh1['cum_ret'][-1]-bench['cum_ret'][-1]}
    performance = performance.append(row, ignore_index=True)
    
    #Returns normalized log_returns of both
    data = prep_features(bench, pgh1)
    
    #Scatter and plot regression line with regression summary
    graph2(data, title=f'Regression of PACT with {i}')
    

for i in list(competitors):
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interval='1d')
    #Calculate returns, cum_ret
    bench = calcs(bench.loc[bench.index>=pandemic_bottom])
    
    #Save data to two dfs
    col_name = i+'_log_return'
    col_name1 = i+'_cum_ret'
    prices[col_name] = bench.log_return
    cum_ret[col_name1] = bench.cum_ret
    
    #Add overperformance of PGH relative
    row = {'over': i, 
           'value': pgh1['cum_ret'][-1]-bench['cum_ret'][-1]}
    performance = performance.append(row, ignore_index=True)
    
    #Returns normalized log_returns of both
    data = prep_features(bench, pgh1)
    
    #Scatter and plot regression line with regression summary
    graph2(data, title=f'Regression of PACT with {i}')
        
for i in list(currencies):
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interval='1d')
    #Calculate returns, cum_ret
    bench = calcs(bench.loc[bench.index>=pandemic_bottom])
    
    #Save data to two dfs
    col_name = i+'_log_return'
    col_name1 = i+'_cum_ret'
    prices[col_name] = bench.log_return
    cum_ret[col_name1] = bench.cum_ret
    
    #Add overperformance of PGH relative
    row = {'over': i, 
           'value': pgh1['cum_ret'][-1]-bench['cum_ret'][-1]}
    performance = performance.append(row, ignore_index=True)
    
    #Returns normalized log_returns of both
    data = prep_features(bench, pgh1)
    
    #Scatter and plot regression line with regression summary
    graph2(data, title=f'Regression of PACT with {i}')
    
dfi.export(performance,"plots/performance.png")

prices = prices.dropna()
cum_ret = cum_ret.dropna()

fig, axs = plt.subplots(1,1,figsize=(12,8))
axs.grid(alpha=0.5, linestyle='dashed', zorder=-1)
for c in cum_ret.columns:
    y = cum_ret[c].values
    # y = y/y.max()
    axs.plot(prices.index, y, label=c)

axs.legend()
axs.set_title('All comparable cumulative returns since '+pandemic_bottom)
path = "plots/cum_returns.png"
# if not os.path.exists(path):
fig.savefig(path, format='png')
fig.show()

#Regression of pgh,asx200
pgh = prices['log_return']
asx = prices['^AXJO_log_return']

lm = sm.OLS(pgh, asx)
res = lm.fit()
pred = res.predict(asx)

with open('summary.txt', 'w') as fh:
    fh.write(res.summary().as_text())

print(res.summary())
    
reg = LinearRegression().fit(asx.values.reshape(-1,1), pgh.values.reshape(-1,1))
pred = reg.predict(asx.values.reshape(-1,1))    
score = reg.score(asx.values.reshape(-1,1), pgh.values.reshape(-1,1))
reg.coef_

























































