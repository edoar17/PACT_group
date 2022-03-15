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
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


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
    axs[0].text(0.15, 0.9, f'Cumulative Return is: {df.cum_ret[-1].round(4)} \n Coefficient is {reg.coef_.round(4)}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[0].transAxes, bbox=props)
    colors = ['g' if i>=0 else 'r' for i in y1]
    axs[1].bar(x, y1, color=colors)
    axs[1].set_title('Cumulative Return')
    
    fig.savefig(f"plots/{title}.png", format='png')
    fig.show()
    

pgh = calcs(pgh)
graph1(pgh, 'Price of PGH.AX')

# From pandemic low
pandemic_bottom = '2020-03-25'
pgh1 = calcs(pgh.loc[pgh.index>=pandemic_bottom])
graph1(pgh1, 'Price of PGH.AX from pandemic low')

# Check relevant ETFs
indexes = {'EWA': 'iShares MSCI-Australia ETF',
           '^AXJO': 'ASX 200',
           'FAIR.AX': 'Betashares Australian Sustainability Leaders ETF ',
           'MVS.AX': 'VanEck Vectors Small Companies Masters ETF'}

currencies = {'AUDUSD=X': 'AUD/USD',
              'AUDJPY=X': 'AUD/JPY'}

                # PPG.AX 800M
competitors = {'PPG.AX': 'Pro-Pac Packaging Limited', #101M
               'ORA.AX': 'Orora Limited', #3B
               'AMC.AX': 'Amcor plc'} #23B

#Linear regressions
def prep_features(df, df2):
    df = calcs(df.loc[df.index>=pandemic_bottom])
    data = df[['log_return']].join(df2['log_return'], rsuffix='_pact')
    data = data.dropna()
    return data

# plot returns
def graph2(df, title):
    x = np.array(df.iloc[:,0]).reshape(-1, 1)
    y = np.array(df.iloc[:,1]).reshape(-1, 1) #Pact is the y

    reg = LinearRegression().fit(x, y)
    pred = reg.predict(x)    
    score = reg.score(x, y)

    rango = [y.min(), y.max()]

    fig, axs = plt.subplots(1,1, figsize=(12,8))
    axs.grid(alpha=0.5, linestyle='dashed', zorder=-1)
    axs.scatter(x,y)
    axs.plot(x, pred, color='r')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs.text(0.15, 0.9, f'R^2 is: {score.round(4)} \n Coefficient is {reg.coef_.round(4)}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=axs.transAxes, bbox=props)
    axs.set_xlim(rango)
    axs.set_ylim(rango)
    axs.set_title(title)
    
    fig.savefig(f"plots/{title}.png", format='png')
    fig.show()


# EWA vs PGH
bench = list(indexes)[0]
bench = yf.Ticker(bench)
bench = bench.history(period='3y', interva='1d')

data = prep_features(bench, pgh1)
graph2(data, title='Regression of PACT with ASX')

for i in list(indexes)[1:]:
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interva='1d')
    
    data = prep_features(bench, pgh1)
    graph2(data, title=f'Regression of PACT with {i}')
    

for i in list(competitors):
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interva='1d')
    
    data = prep_features(bench, pgh1)
    graph2(data, title=f'Regression of PACT with {i}')
        
for i in list(currencies):
    bench = yf.Ticker(i)
    bench = bench.history(period='3y', interva='1d')
    
    data = prep_features(bench, pgh1)
    graph2(data, title=f'Regression of PACT with {i}')




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    























































