#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 23:33:44 2022

@author: edoar17
"""

import yfinance as yf
import pandas as pd
import numpy as np

def calc_return(series):
    return (series/series.shift(1))-1
    
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











