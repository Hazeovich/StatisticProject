import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from zigzag import *


symbol = 'symbol=AAPL'
interval='interval=60min'
PATH_TO_CSV = 'data/' + symbol + interval + '.csv'

df = pd.read_csv(PATH_TO_CSV)
df.columns=['point', 'date/time', 'open', 'hight', 'low', 'close', 'volume']
df['date/time'] = pd.to_datetime(df.loc[:,df.columns[1]])
df.sort_values('date/time', ascending=True, inplace=True)
df.reset_index(inplace=True)
df = df.iloc[:,2:]

volumetoplt = df[['date/time', 'volume']]
volumetoplt.set_index('date/time', inplace=True)

toplotdf = df.iloc[:,:5]
toplotdf.set_index('date/time', inplace=True)

df_open = df[['date/time', 'open']]
df_open.set_index(df_open.columns[0], inplace=True)

df_hight = df[['date/time', 'hight']]
df_hight.set_index(df_hight.columns[0], inplace=True)

df_low = df[['date/time', 'low']]
df_low.set_index(df_low.columns[0], inplace=True)

df_close = df[['date/time', 'close']]
df_close.set_index(df_close.columns[0], inplace=True)

  
def showPlot(asset = pd.DataFrame()):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    
    indicator_asset = asset[(asset[asset.columns[1]] != 0) & (asset[asset.columns[1]] != 2)][asset.columns[0]]
    
    ax.plot(asset[asset.columns[0]], color='red', linewidth = 1)
    ax.plot(indicator_asset, linewidth = 2, linestyle = '--')
    
    
    
    asset_down = asset[asset[asset.columns[1]] == -1][asset.columns[0]]
    asset_up = asset[asset[asset.columns[1]] == 1][asset.columns[0]]
    asset_non = asset[asset[asset.columns[1]] == 2][asset.columns[0]]
    
    ax.scatter(asset_down.index, asset_down.values, color='r')
    ax.scatter(asset_up.index, asset_up.values, color='g')
    ax.scatter(asset_non.index, asset_non.values, color='magenta')
    
    ax.figure.legend(['График активов', 'ZigZag', 'Верхний разворот', 'Нижний разворот', 'Нейтраль'])
    
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

def getDistribution(asset = pd.DataFrame(), show_plt = True):
    asset_pivots = asset[(asset[asset.columns[1]] != 0) & (asset[asset.columns[1]] != 2)]
    
    time_diff = pd.DataFrame()
    for i in range(len(asset_pivots.index)-1):
        date1 = asset_pivots.iloc[i,:].name
        date2 = asset_pivots.iloc[i+1,:].name
        temp_time = pd.DataFrame(data=[date2-date1])
        time_diff = pd.concat([time_diff, temp_time])
    time_diff.reset_index(inplace=True)
    time_diff = time_diff.iloc[:,1]
    
    h = time_diff.astype('timedelta64[h]').values
    h = h.astype(np.float64) #Тут точки для гистограммы
    
    pd.DataFrame(h).to_csv('histagram/points.csv') 
    
    counts, bins = np.histogram(h, bins=20, density=False)
    hist_df = pd.DataFrame(data=[counts, bins])
    hist_df = hist_df.T
    os.makedirs('histagram', exist_ok=True)
    hist_df.to_csv(f'histagram/hist_{symbol}_{interval}_{asset.columns[0]}.csv')
    
    indexmax = counts.argmax()
    if show_plt:
        plt.hist(bins[:-1], bins, weights=counts)
        plt.xlabel('Интервал между разворотными точками в минутах')
        plt.ylabel('Кол-во разворотных точек')
        plt.title('Гистограма распределения интервалов между разворотными точками')
        plt.style.use("seaborn-v0_8-pastel")
        plt.show()
    return round(np.average(h)/2)
  
def getPrehistory(asset = pd.DataFrame(), time_history = int()):
    points = asset[asset[asset.columns[1]] != 0]
    prehistory = pd.DataFrame()
    for i in reversed(points.index):
        temparr = asset.loc[:i].tail(time_history)[asset.columns[0]].values
        indicator = points.loc[i, points.columns[1]]
        
        if indicator == 2: indicator = 0
        linedf = []
        linedf.append(indicator)
        for i in temparr: linedf.append(i)
        
        if len(temparr) == time_history:
            temparr = [linedf]
            tempdf = pd.DataFrame(temparr)
            prehistory = pd.concat([prehistory, tempdf])
    
    prehistory.reset_index(inplace=True)
    prehistory = prehistory.iloc[:,1:] 
    prehistory.rename(columns={0:'indicator'}, inplace=True)
    print(prehistory)
    return prehistory
        
def zigzagIndicator(h, asset = pd.DataFrame(), show_plt = False, show_hist = False):
    val_name = asset.columns[0]
    pivots = peak_valley_pivots(asset[val_name].values, h, -h) #?Добрые люди сделали zigzag за меня а я нагло украл (возвращает numpy.ndarray)
    pivots *= -1
    
    firstidx = 0
    lastidx = 0
    for i in range(len(pivots)):
        if pivots[i] != 0:
            lastidx = firstidx
            firstidx = i
            if i != 0:
                hoursdiff = firstidx-lastidx
                coef = random.betavariate(alpha=2, beta=3)
                idx_zeromark = round(coef*hoursdiff)+lastidx
                pivots[idx_zeromark] = 2
            
    asset.insert(1, "updown", pivots)
    
    if show_plt: showPlot(asset)
    
    time_history = getDistribution(asset, show_hist)
    asset_indicators = asset[asset[asset.columns[1]] != 0]
    print(f"Time history len: {time_history}h\nCount of indicator's point: {asset_indicators.shape[0]}")
    prehistory = getPrehistory(asset=asset, time_history=time_history)
    os.makedirs('prehistory', exist_ok=True)
    prehistory.to_csv(f'prehistory\dataset_{symbol}_{interval}_{asset.columns[0]}.csv') #?Сохраняем в csv
          
h_open = 0.023
h_hight = 0.02
h_low = 0.02
h_close = 0.08    
zigzagIndicator(h_close, df_close, True, True)
# zigzagIndicator(h_open, df_open, True, True)
# zigzagIndicator(h_hight, df_hight, True, True)
# zigzagIndicator(h_low, df_low, True, True)
# zigzagIndicator(h_close, df_close, True, True)