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

def to_monthly(df, last_day=False):
    if last_day==False:
        out = df.groupby([df.index.year, df.index.month]).head(1)
    else:
        out = df.groupby([df.index.year, df.index.month]).tail(1)
    return out
 
def calcs(df, floor):
    if not floor:
        pass
    elif floor=='monthly':
        df['year'] = df.index.year
        df['quarter'] = df.index.month
        df = df.sort_values(['year', 'month']).drop_duplicates(['year', 'month'])
    elif floor=='quarter':
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df = df.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
    elif floor=='semi':    
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df = df.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
        df = df.loc[(df.quarter==2) | (df.quarter==4)]
        # print(df)
    
        
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
    # if not os.path.exists(path):
    fig.savefig(f"plots/{title}.png", format='png')
    fig.show()

   

# Download PGH data
ticker = 'PGH.AX'

PGH_ = yf.Ticker(ticker)
pgh = PGH_.history(period='5y', interval='1d')
pgh = calcs(pgh, False)
graph1(pgh, 'Price of PGH.AX')

# From pandemic low
pandemic_bottom = '1900-01-01'
# pgh = calcs(pgh.loc[pgh.index>=pandemic_bottom])
# # graph1(pgh1, 'Price of PGH.AX from pandemic low')

# Check relevant ETFs
indexes = {'EWA': 'iShares MSCI-Australia ETF',
           '^AXJO': 'ASX 200',
           'FAIR.AX': 'Betashares Australian Sustainability Leaders ETF ',
           'MVS.AX': 'VanEck Vectors Small Companies Masters ETF',
           'DBC': 'Invesco DB Commodity Index Tracking Fund',
           'DBO': 'Invesco DB Oil Fund',
           #Sustainable ETFs
           'ERTH':'Invesco MSCI Sustainable Future ETF '}

currencies = {'AUDUSD=X': 'AUD/USD',
              'AUDJPY=X': 'AUD/JPY',
              'AUDNZD=X': 'AUD/NZD',
              'AUDHKD=X': 'AUD/HKD',
              'AUDCNY=X': 'AUDCNY'}

                # PPG.AX 800M
competitors = {'PPG.AX': 'Pro-Pac Packaging Limited', #101M
               'ORA.AX': 'Orora Limited', #3B
               'AMC.AX': 'Amcor plc'} #23B


#Linear regressions
def prep_features(df, df2):
    # df = df.loc[df.index>=pandemic_bottom]
    data = df[['log_return']].join(df2['log_return'], rsuffix='_pact') 
    data = data.dropna()
    # data = to_monthly(data, last_day=(False))
    # print(data)
    data['log_return'] = data['log_return']/data['log_return'].max()
    data['log_return_pact'] = data['log_return_pact']/data['log_return_pact'].max()
    
    # print(data)
    return data

# plot returns
def graph2(df, title, folder):
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

    rango = [-1, 1]

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

    path = f"{folder}/{title}.png"
    # if not os.path.exists(path):
    fig.savefig(path, format='png')
    fig.show()
    


def do_plots(pgh, indexes, competitors, currencies, folder, interval, floor, prices, cum_ret):
    # Add performance
    performance = pd.DataFrame(columns=['over', 'value'])
    # prices = pgh[['log_return']]
    # cum_ret = pgh[['cum_ret']]   
    
    for i in list(indexes):
        bench = yf.Ticker(i)
        bench = bench.history(period='5y', interval=interval)
        #Calculate returns, cum_ret
        bench = calcs(bench.loc[bench.index>=pandemic_bottom], floor)
        
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
        data = prep_features(bench, pgh)
        
        #Scatter and plot regression line with regression summary
        graph2(data, title=f'Regression of PACT with {i} - {floor}', folder=folder)
        
    
    for i in list(competitors):
        bench = yf.Ticker(i)
        bench = bench.history(period='5y', interval=interval)
        #Calculate returns, cum_ret
        bench = calcs(bench.loc[bench.index>=pandemic_bottom], floor)
        
        #Save data to two dfs
        col_name = i+'_log_return'
        col_name1 = i+'_cum_ret'
        prices[col_name] = bench.log_return
        cum_ret[col_name1] = bench.cum_ret
        
        #Add overperformance of PGH relative
        row = {'over': i, 
               'value': pgh['cum_ret'][-1]-bench['cum_ret'][-1]}
        performance = performance.append(row, ignore_index=True)
        
        #Returns normalized log_returns of both
        data = prep_features(bench, pgh)
        
        #Scatter and plot regression line with regression summary
        graph2(data, title=f'Regression of PACT with {i} - {floor}', folder=folder)
            
    for i in list(currencies):
        bench = yf.Ticker(i)
        bench = bench.history(period='5y', interval=interval)
        #Calculate returns, cum_ret
        bench = calcs(bench.loc[bench.index>=pandemic_bottom], floor)
        # print(bench)
        #Save data to two dfs
        col_name = i+'_log_return'
        col_name1 = i+'_cum_ret'
        prices[col_name] = bench.log_return
        cum_ret[col_name1] = bench.cum_ret
        
        #Add overperformance of PGH relative
        row = {'over': i, 
               'value': pgh['cum_ret'][-1]-bench['cum_ret'][-1]}
        performance = performance.append(row, ignore_index=True)
        
        #Returns normalized log_returns of both
        data = prep_features(bench, pgh)
        
        #Scatter and plot regression line with regression summary
        graph2(data, title=f'Regression of PACT with {i} - {floor}', folder=folder)
        
    dfi.export(performance,f"{folder}/performance.png")

pgh1 = calcs(pgh, floor='semi')
prices_s = pgh1[['log_return']]
cum_ret_s = pgh1[['cum_ret']] 
do_plots(pgh1, indexes, competitors, currencies, folder='plots_semi', interval='1d', floor='semi',
         prices=prices_s, cum_ret=cum_ret_s)

pgh1 = calcs(pgh, floor='quarter')
prices_q = pgh1[['log_return']]
cum_ret_q = pgh1[['cum_ret']]   
do_plots(pgh1, indexes, competitors, currencies, folder='plots_quarterly', interval='1d', floor='quarter',
         prices=prices_q, cum_ret=cum_ret_q)

pgh1 = calcs(pgh, floor=False)
prices_d = pgh1[['log_return']]
cum_ret_d = pgh1[['cum_ret']]   
do_plots(pgh1, indexes, competitors, currencies, folder='plots', interval='1d', floor=False, 
         prices=prices_d, cum_ret=cum_ret_d)

prices = prices.dropna()
cum_ret = cum_ret.dropna()

prices_q.isna().sum()
prices_q.dropna()

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

#correlation matrix
corrM = prices_q.corr()
sns.heatmap(corrM)
# dfi.export(corrM, "correlation_matrix.png")
# corrM.to_csv('correlation_matrix.csv')

# #inflation relation to share price
# with open('data/inflation.csv', mode='r') as f:
#     inflation = pd.read_csv(f)
# inflation = inflation.drop(columns='Unnamed: 0')

# with open('data/PGH.AX_quarterly', mode='r') as f:
#     quarterly = pd.read_csv(f)

# quarterly = inflation.join(quarterly, rsuffix='_inf').dropna()
# quarterly1 = quarterly[['Date', 'return', 'pct_change']]

# lm = sm.OLS(quarterly['return'], quarterly['pct_change'])
# reg = lm.fit()
# reg.summary()

# graph2(quarterly1.drop(columns='Date'), title='Regr', folder='plots_quarterly')

prices_currencies = prices_q[prices_q.columns[prices_q.columns.str.contains('=')]]
prices_currencies['log_return'] = prices_q['log_return']
prices_currencies = prices_currencies.dropna()

lm = sm.OLS(prices_currencies['log_return'], exog=prices_currencies.drop('log_return', axis=1))
reg = lm.fit()
reg.summary()

# Daily
prices_currencies = prices_q[prices_d.columns[prices_d.columns.str.contains('=')]]
prices_currencies['log_return'] = prices_d['log_return']
prices_currencies = prices_currencies.dropna()

lm = sm.OLS(prices_currencies['log_return'], exog=prices_currencies.drop('log_return', axis=1))
reg = lm.fit()
reg.summary()




































