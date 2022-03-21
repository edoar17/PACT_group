#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 23:33:44 2022

@author: edoar17
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('PACT/')

def calc_return(series):
    return (series/series.shift(1))-1

def log_return(series):
    return np.log(series).diff()
    
# Australia inflation
url = 'https://www.rateinflation.com/consumer-price-index/australia-historical-cpi/'
html = pd.read_html(url)
inflation = html[0].loc[html[0].Year>=2017]
inflation = inflation.drop('Annual', axis=1)
inflation.columns = ['Year', 1, 2, 3, 4]
inflation = pd.melt(inflation, id_vars='Year', value_vars=inflation.columns[1:], 
                    var_name='quarter', value_name='value')
inflation = inflation.set_index(['Year', 'quarter']).sort_index(ascending=True).reset_index()

inflation['pct_change'] = (inflation['value']/inflation['value'].shift(1))-1
inflation = inflation.dropna()
#Save to csv
inflation.to_csv('data/inflation.csv')

# Quarterly ASX200 and PGH.AX
ASX_ = yf.Ticker('^AXJO')
asx = ASX_.history(period='5y', interval='3mo')
asx['return'] = calc_return(asx['Close'])
asx.to_csv('data/ASX_quarterly')

PGH_ = yf.Ticker('PGH.AX')
pgh = PGH_.history(period='5y', interval='1d')
pgh['year'] = pgh.index.year
pgh['quarter'] = pgh.index.quarter
pgh = pgh.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
pgh['return'] = calc_return(pgh['Close'])
pgh = pgh.dropna()
pgh.to_csv('data/PGH.AX_quarterly')

# Semi-annually ASX200 and PGH.AX
asx = ASX_.history(period='5y', interval='3mo')
asx['year'] = asx.index.year
asx['quarter'] = asx.index.quarter
asx = asx.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
asx = asx.loc[(asx.quarter==2) | (asx.quarter==4)]
asx['return'] = calc_return(asx['Close'])
asx = asx.dropna()
asx.to_csv('data/ASX_semi')

pgh = PGH_.history(period='5y', interval='1d')
pgh['year'] = pgh.index.year
pgh['quarter'] = pgh.index.quarter
pgh = pgh.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
pgh = pgh.loc[(pgh.quarter==2) | (pgh.quarter==4)]
pgh['return'] = calc_return(pgh['Close'])
pgh = pgh.dropna()
pgh.to_csv('data/PGH.AX_semi')

# Build AUD Index attribute weight of Australian Trade
currencies = {'AUDUSD=X': 'United States',
              'AUDJPY=X': 'Japan',
              'AUDNZD=X': 'New Zealand',
              'AUDSGD=X': 'Singapore',
              'AUDHKD=X': 'Hong Kong',
              'AUDCNY=X': 'China'}

currencies = {'United States':'AUDUSD=X',
              'Japan':'AUDJPY=X',
              'New Zealand':'AUDNZD=X',
              'Singapore':'AUDSGD=X',
              'Hong Kong':'AUDHKD=X',
              'China':'AUDCNY=X'}

url = 'https://www.dfat.gov.au/publications/trade-and-investment/trade-and-investment-glance-2020'
trade = pd.read_html(url)[3]
trade = trade[['Trading partners(a)(b)', '% share', 'Total']]
trade = trade.rename(columns = {'Trading partners(a)(b)':'country',
                                '% share': 'share'})
trade = trade.iloc[:10,:]
trade['currency'] = trade['country'].apply(lambda x: currencies.get(x) if x in currencies else 'AUDUSD=X')
trade['currency'][trade['country']=='Thailand'] = 'AUDSGD=X'
trade['currency'][trade['country']=='Malaysia'] = 'AUDSGD=X'
trade['share'] = trade['share'].astype('float')
trade['Total'] = trade['Total'].astype('float')

trade_curr = trade.groupby('currency')['Total'].agg(sum)
trade_curr = trade_curr/trade_curr.sum()

returns = pd.DataFrame()

for i in currencies:
    ticker = currencies.get(i)
    if ticker in trade_curr:
        TICK = yf.Ticker(ticker)
        ret = TICK.history(period='5y', interval='1d', actions=False)
        
        ret['log_return'] = log_return(ret['Close']) * trade_curr[ticker]
        # ret['cum_ret'] = np.exp(ret['log_return'].cumsum())
        
        logret = 'log_return_' + i 
        # cumret = 'cum_return_' + i
        
        if returns.empty:
            returns['AUD_weighted'] = ret['log_return']
        else:
            returns['AUD_weighted'] = ret['log_return'] + returns['AUD_weighted']
    
returns['cumret'] = np.exp(returns['AUD_weighted'].cumsum())
# returns['cumret'] = 1/returns['cumret_inv']
returns.dropna()
returns.to_csv('data/AUD_strength_daily')

sns.lineplot(x=returns.index, y=returns['cumret'])
# plt.savefig('data/aud_strength_daily.png')

# Save quarterly 
returns['year'] = returns.index.year
returns['quarter'] = returns.index.quarter
returns_quarterly = returns.sort_values(['year', 'quarter']).drop_duplicates(['year', 'quarter'])
returns_quarterly = returns_quarterly.dropna()
# returns_quarterly['cumret'].to_csv('data/AUD_strength_quarterly')

# and semi annually
returns_semi = returns_quarterly.loc[(returns_quarterly.quarter==2) | (returns_quarterly.quarter==4)]
returns_semi.to_csv('data/AUD_strength_semi')
# sns.lineplot(x=returns_semi.index, y=returns_semi.cumret)






























