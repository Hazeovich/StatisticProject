import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
import os  
import datetime as dt
from zigzag import *

###################################################

def getDaysDifference(day1, day2):
    days_str = str(dt.datetime.strptime(day1, '%Y-%m-%d') - dt.datetime.strptime(day2, '%Y-%m-%d'))
    try: 
        days_int = int(days_str[:days_str.find(' ')])
        return days_int
    except:
        days_int = 0
        return days_int

def zigzag_indicator(X, h):
    pivots = peak_valley_pivots(X, h, -h)
    return pivots

###################################################

path_data = 'data/out.csv'
tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
ticker = tickers_list[0]

data = pd.DataFrame()

try: 
    data = pd.read_csv(path_data)
    data = data.set_index('Date')
    
except:
    data.rename(columns=[ticker])
    data[ticker] = yf.download(ticker,'2016-01-01','2019-08-01')['Adj Close']
    os.makedirs('data', exist_ok=True)  
    data.to_csv(path_data)

minVal = data.min()[ticker]
maxVal = data.max()[ticker]
h = (maxVal - minVal)/maxVal * 100
h = 0.05

pivots = zigzag_indicator(data[ticker].values, h)
pds_pivots = pd.Series(data=data[ticker], index=data.index)
pds_pivots = pds_pivots[pivots != 0]

dt_data = []
for i in data.index: dt_data.append(dt.datetime.fromisoformat(i))
dt_pivots = []
for i in pds_pivots.index: dt_pivots.append(dt.datetime.fromisoformat(i))

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

ax.plot(dt_data, data[ticker].values, color = 'r', linewidth = 1)
ax.plot(dt_pivots, pds_pivots.values, linewidth = 2, linestyle='--')
ax.scatter(np.array(dt_data)[pivots==1], data[ticker].values[pivots==1], color='r')
ax.scatter(np.array(dt_data)[pivots==-1], data[ticker].values[pivots==-1], color='g')

ax.set_xlim(ax.get_xlim())
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_minor_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%y-%m-%d'))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))


fig.set_figwidth(12)
fig.set_figheight(5)

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()