import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#Список компаний
tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']

#Создаем Dataframe для последующего хранения данных
data = pd.DataFrame(columns=tickers_list)

#Загружаем все данные
for ticker in tickers_list:
    data[ticker] = yf.download(ticker,'2016-01-01','2019-08-01')['Adj Close']

# Вывод первых 5ти строк Dataframe
print(data.head())

#Строим график 
plt.figure()
plt.plot(data)

plt.legend(tickers_list)

plt.title("Daily", fontsize=16)
plt.xlabel("Price", fontsize=14)
plt.ylabel("Year", fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

plt.show()