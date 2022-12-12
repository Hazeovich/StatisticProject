import pandas as pd
import requests
import csv
import time

link = 'https://www.alphavantage.co/query?'
fucn = 'function=TIME_SERIES_INTRADAY_EXTENDED'
symbol = 'symbol=AAPL'
interval='interval=60min'
apikey='apikey=104NFNEMPQ6Q3XL9'
PATH_TO_CSV = 'D:/Programing/Project/data/' + symbol + interval + '.csv'
slice_data = []
years = 1
for year in range(1,2+1):
    for month in range(1, 12+1):
        slice_data.append(f'slice=year{year}month{month}')

df = pd.DataFrame()

for period in slice_data:
    API_URL = link+"&"+fucn+"&"+symbol+"&"+interval+"&"+period+"&"+apikey
    with requests.Session() as s:
        download = s.get(API_URL)
        print(f"the status code is:{download.status_code} for {period}")
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        temp_df = pd.DataFrame(my_list)
        temp_df = temp_df.iloc[1:]
        df = pd.concat([df, temp_df])
    time.sleep(20)
    
df.to_csv(PATH_TO_CSV)