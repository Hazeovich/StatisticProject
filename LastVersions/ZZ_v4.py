import datetime as dt
import os
import random as rnd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from zigzag import *

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

def getBettaRandomDate(date1, date2, asset, alpha, betta):
    coef = rnd.betavariate(alpha, betta)
    frame = asset.loc[date1:date2]
    date = frame.iloc[round(len(frame.index.values)*coef)]
    return date.name

def converToFrameForm(pd_series, col_name, start_index=0):
    idx = start_index
    frames = pd.DataFrame()
    for i in pd_series.index:
        tmp_df = pd.DataFrame(data={'date':[i],'updown':[-1 * idx], col_name:pd_series.loc[i]})
        tmp_df.set_index('date', inplace=True)
        frames = pd.concat([frames, tmp_df])
        idx = -1 * idx
    return frames

def getPrehistory(points, asset, time_history):
    frames_time_history = pd.DataFrame()
    frame_counter = len(points.index)
    col_name = asset.columns.values[0]
    for i in reversed(points.index):
        frame_prehistory = asset.loc[:i].tail(time_history)
        tmp_arr = [[],[],[]]
        tmp_val = []
        for j in reversed(frame_prehistory.index):
            tmp_arr[0].append(frame_counter)
            tmp_arr[1].append(points.loc[i]['updown'])
            tmp_arr[2].append(j)
            tmp_val.append(frame_prehistory.loc[j][col_name])
        tmp_index = pd.MultiIndex.from_arrays(tmp_arr, names=('frame','updown', 'date'))    
        tmp_df = pd.DataFrame(data=tmp_val, index=tmp_index)
        frames_time_history = pd.concat([frames_time_history, tmp_df])
        frame_counter -= 1
    frames_time_history.sort_index(inplace=True)
    return frames_time_history
        
def zigzagIndicator(h, time_history, show_plt, asset):
    col_name = asset.columns.values[0]
    pivots = peak_valley_pivots(asset[col_name].values, h, -h) #?Добрые люди сделали zigzag за меня а я нагло украл (возвращает numpy.ndarray)
    pds_pivots = pd.Series(data=asset[col_name], index=asset.index) 
    pds_pivots = pds_pivots[pivots != 0] #?Выкидываем все элементы asset, которые не являются разворотной точкой
    pivots_date_arr = pds_pivots.index.values
    
    zero_mark_points = []
    for i in range(len(pds_pivots.index)-1):
        date_start = pivots_date_arr[i]
        date_end = pivots_date_arr[i+1]
        new_date = getBettaRandomDate(date1=date_start, date2=date_end, asset=asset, alpha=3, betta=3)
        zero_mark_points.append(new_date)
    
    pds_zeros = pd.Series(data=asset[col_name], index=asset.index)
    pds_zeros = pds_zeros[zero_mark_points]
    
    dt_data = converToDt(asset.index) #?преобразуем время из текста в объект класса datatime для того чтобы граффик красиво выглядел
    dt_pivots = converToDt(pds_pivots.index)
    dt_zeros = converToDt(pds_zeros.index)
    
    pivots_frames = converToFrameForm(pds_pivots, col_name, pivots[0])
    zeros_frames = converToFrameForm(pds_zeros, col_name)
    all_points = pd.concat([pivots_frames, zeros_frames])
    all_points.sort_index(inplace=True)
    frames_time_history = getPrehistory(all_points, asset, time_history)
    
    os.makedirs('prehistory', exist_ok=True)
    frames_time_history.to_csv(f'D:\Programing\Project\prehistory\out_{col_name}.csv') #?Сохраняем в csv
        
    if show_plt == True:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

        ax.plot(dt_data, asset[col_name].values, color = 'r', linewidth = 1)
        ax.plot(dt_pivots, pds_pivots.values, linewidth = 2, linestyle='--')
        ax.scatter(np.array(dt_data)[pivots==1], asset[col_name].values[pivots==1], color='r')
        ax.scatter(np.array(dt_data)[pivots==-1], asset[col_name].values[pivots==-1], color='g')
        ax.scatter(np.array(dt_zeros), pds_zeros.values, color='magenta')

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

path_data = 'D:\Programing\Project\data\out.csv'
tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
ticker = tickers_list[0]

data = pd.DataFrame()

try: 
    data = pd.read_csv(path_data)
    data = data.set_index('Date')
    print(data)
except:
    data.rename(columns=[ticker])
    data[ticker] = yf.download(ticker,'2016-01-01','2019-08-01')['Adj Close']
    os.makedirs('data', exist_ok=True)  
    data.to_csv(path_data)
    
    
h = 0.05
time_history = zigzagIndicator(h, 10, True, data)
#print(time_history.head(30)) 