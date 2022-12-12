import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
import os  
import datetime as dt
from zigzag import *
import random as rnd

#!##################################################

def getDaysDifference(day1, day2):
    days_str = str(dt.datetime.strptime(day1, '%Y-%m-%d') - dt.datetime.strptime(day2, '%Y-%m-%d'))
    try: 
        days_int = int(days_str[:days_str.find(' ')])
        return days_int
    except:
        days_int = 0
        return days_int

def converToDt(dates):
    dt_dates = []
    for i in dates: dt_dates.append(dt.datetime.fromisoformat(i))
    return dt_dates

def zigzagIndicator(h, time_history, show_plt, asset):
    frames_time_history = pd.DataFrame() #?Сюда сохраняются фреймы с предыстрией
    
    col_name = asset.columns.values[0]
    pivots = peak_valley_pivots(asset[col_name].values, h, -h) #?Добрые люди сделали zigzag за меня а я нагло украл (возвращает numpy.ndarray)
    pds_pivots = pd.Series(data=asset[col_name], index=asset.index) 
    pds_pivots = pds_pivots[pivots != 0] #?Выкидываем все элементы asset, которые не являются разворотной точкой
    
    dt_data = converToDt(asset.index) #?преобразуем время из текста в объект класса datatime для того чтобы граффик красиво выглядел
    dt_pivots = converToDt(pds_pivots.index)
    
    frame_counter = len(pds_pivots.index)
    pivots_short = pivots[pivots != 0]
    
    for i in reversed(pds_pivots.index):
        frame_prehistory = asset.loc[:i].tail(time_history) #?Берем дату разворотной точки и получаем кусок в time_history дней
        tmp_arr = [[],[],[]]
        tmp_val = []
        for i in reversed(frame_prehistory.index):
            tmp_arr[0].append(frame_counter)
            tmp_arr[1].append(-1 * pivots_short[frame_counter-1])
            tmp_arr[2].append(i)
            tmp_val.append(frame_prehistory.loc[i][col_name])
        tmp_index = pd.MultiIndex.from_arrays(tmp_arr, names=('frame','updown', 'date'))    
        tmp_df = pd.DataFrame(data=tmp_val, index=tmp_index)
        frames_time_history = pd.concat([frames_time_history, tmp_df])
        frame_counter -= 1
    
    frames_time_history.sort_index(ascending=True, inplace=True) #?Сортируем по индексам frame чтобы поряд был от 1,2,3,...
    os.makedirs('prehistory', exist_ok=True)
    frames_time_history.to_csv(f'prehistory/out_{col_name}.csv') #?Сохраняем в csv
        
    if show_plt == True:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

        ax.plot(dt_data, asset[col_name].values, color = 'r', linewidth = 1)
        ax.plot(dt_pivots, pds_pivots.values, linewidth = 2, linestyle='--')
        ax.scatter(np.array(dt_data)[pivots==1], asset[col_name].values[pivots==1], color='r')
        ax.scatter(np.array(dt_data)[pivots==-1], asset[col_name].values[pivots==-1], color='g')

        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator(bymonth=range(1,12,2)))
        ax.xaxis.set_major_formatter(DateFormatter('%y-%m-%d'))

        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

        fig.set_figwidth(12)
        fig.set_figheight(5)

        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.show()
    
    return frames_time_history

#!##################################################

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
    
h = 0.08
    
time_history = zigzagIndicator(h, 10, True, data)
print(time_history.head(30))