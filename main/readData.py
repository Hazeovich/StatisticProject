import pandas as pd

class ReadData:
    def __init__(self, path):
        self.path = path
        self.df = None
    
    def read_csv_to_pd(self):
        self.df = pd.read_csv(self.path)
        self.df = self.df.iloc[:,1:]
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
    
    def get_close_values(self):
        return self.df[['datetime', 'close']].set_index('datetime')
    
if __name__ == '__main__':
    data = ReadData('data/EURUSD_MIN.csv')
    data.read_csv_to_pd()
    print(data.df)
    