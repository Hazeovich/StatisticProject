import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os  
import datetime as dt


def getDaysDifference(day1, day2):
    days_str = str(dt.datetime.strptime(day1, '%Y-%m-%d') - dt.datetime.strptime(day2, '%Y-%m-%d'))
    try: 
        days_int = int(days_str[:days_str.find(' ')])
        return days_int
    except:
        days_int = 0
        return days_int

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

data = data.reset_index()
print(data.head())


minVal = data.min()[ticker]
maxVal = data.max()[ticker]
h = (maxVal - minVal)/maxVal * 100
h = 2

return_points = pd.DataFrame([data.loc[0].values], index=[0], columns=['Date', ticker])

start_point = data.loc[0]
end_point = start_point
max_diff_point = start_point
time_history = 30

i = 0
max_difference = 0
while end_point.Date != data.loc[len(data)-1].Date:
    
    end_point = data.loc[i]
    if getDaysDifference(end_point.Date, start_point.Date) <= time_history:
        i += 1
        difference = abs(start_point[ticker] - end_point[ticker])
        if difference > max_difference:
            max_difference = difference
            max_diff_point = end_point
    else:
        start_point = max_diff_point
        i = data.index[data.Date==start_point.Date].to_list()[0]
        return_points = pd.concat([return_points, pd.DataFrame([max_diff_point.values], index=[i], columns=['Date', ticker])], ignore_index=True)
        max_difference = 0

clear_rp = pd.DataFrame([return_points.loc[0].values], index=[0], columns=['Date', ticker])
old_diff = 0    
for i in range(len(return_points)-1):
    diff = return_points.loc[i+1][ticker] - return_points.loc[i][ticker]
    if diff > 0:
        diff = 1
    elif diff < 0: 
        diff = -1
    else: 
        diff = 0
    
    if diff != old_diff and abs(return_points.loc[i+1][ticker] - return_points.loc[i][ticker]) >= h:
        clear_rp = pd.concat([clear_rp, pd.DataFrame([return_points.loc[i].values], index=[i], columns=['Date', ticker])], ignore_index=True)
    
    old_diff = diff
    
    # if getDaysDifference(end_point.Date, start_point.Date) <= time_history:
    #     i += 1
    #     difference = start_point[ticker] - end_point[ticker]
    #     if abs(difference) >= h and abs(difference) > max_difference:
    #         max_difference = abs(difference)
    #         max_diff_point = end_point
    # else:
    #     if max_diff_point.Date != end_point.Date:
    #         return_points = pd.concat([return_points, pd.DataFrame([max_diff_point.values], index=[i], columns=['Date', ticker])], ignore_index=True)
    #         start_point = max_diff_point
    #     else:
    #         new_index = data.index[data.Date==start_point.Date].to_list()[0] + 1
    #         start_point = data.loc[new_index]
    #         end_point = start_point
        
    
print(return_points)    

# for i in range(len(data)):
    
#     end_point = data.loc[i]
#     days_str = str(dt.datetime.strptime(end_point.Date, '%Y-%m-%d') - dt.datetime.strptime(start_point.Date, '%Y-%m-%d'))
#     try: 
#         days_int = int(days_str[:days_str.find(' ')])
#     except:
#         days_int = 0
    
#     if days_int >= time_history and end_point[ticker] > local_max[ticker]:
#         local_max = end_point
#         if abs(start_point[ticker] - end_point[ticker]) >= h:
#             start_point = end_point
#             return_points = pd.concat([return_points, pd.DataFrame([end_point.values], index=[i], columns=['Date', ticker])], ignore_index=True)
#     if days_int >= time_history and end_point[ticker] < local_min[ticker]:
#         local_min = end_point
#         if abs(start_point[ticker] - end_point[ticker]) >= h:
#             start_point = end_point
#             return_points = pd.concat([return_points, pd.DataFrame([end_point.values], index=[i], columns=['Date', ticker])], ignore_index=True)

fig, ax = plt.subplots()

ax.plot(data.Date.values, data[ticker].values, color = 'r', linewidth = 1)
ax.plot(return_points.Date.values, return_points[ticker].values, linewidth = 0, marker='o')
ax.plot(clear_rp.Date.values, clear_rp[ticker].values, linewidth = 0, marker='o', color='red')

#  Устанавливаем интервал основных делений:
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(150))
#  Устанавливаем интервал вспомогательных делений:
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
#ax.tick_params(labelrotation=90, )

#  Тоже самое проделываем с делениями на оси "y":
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))


fig.set_figwidth(10)
fig.set_figheight(5)

plt.show()