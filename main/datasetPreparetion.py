import pandas as pd
from zigzag import *
import random
import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self, df, timerange):
        self.df = df
        self.timerange = timerange
    
    def showPlot(self):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        
        self.indicator_df = self.df[(self.df[self.df.columns[1]] != 0) & (self.df[self.df.columns[1]] != 2)][self.df.columns[0]]
        
        ax.plot(self.df[self.df.columns[0]], color='red', linewidth = 1)
        ax.plot(self.indicator_df, linewidth = 2, linestyle = '--')
        
        self.df_down = self.df[self.df[self.df.columns[1]] == -1][self.df.columns[0]]
        self.df_up = self.df[self.df[self.df.columns[1]] == 1][self.df.columns[0]]
        self.df_non = self.df[self.df[self.df.columns[1]] == 2][self.df.columns[0]]
        
        ax.scatter(self.df_down.index, self.df_down.values, color='r')
        ax.scatter(self.df_up.index, self.df_up.values, color='g')
        ax.scatter(self.df_non.index, self.df_non.values, color='magenta')
        
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.show()
    
    def getDistribution(self, show_plt):
        self.df_pivots = self.df[(self.df[self.df.columns[1]] != 0) & (self.df[self.df.columns[1]] != 2)]
        
        self.time_diff = pd.DataFrame()
        for i in range(len(self.df_pivots.index)-1):
            self.date1 = self.df_pivots.iloc[i,:].name
            self.date2 = self.df_pivots.iloc[i+1,:].name
            self.temp_time = pd.DataFrame(data=[self.date2-self.date1])
            self.time_diff = pd.concat([self.time_diff, self.temp_time])
        self.time_diff.reset_index(inplace=True)
        self.time_diff = self.time_diff.iloc[:,1]
        
        self.h = self.time_diff.astype('timedelta64[h]').values
        self.h = self.h.astype(np.float64) #Тут точки для гистограммы
        
        self.counts, self.bins = np.histogram(self.h, bins=20, density=False)
        self.hist_df = pd.DataFrame(data=[self.counts, self.bins])
        self.hist_df = self.hist_df.T
        
        if show_plt:
            plt.stairs(self.counts, self.bins)
            plt.show()
        return round(np.average(self.h)/2)
    
    def getPrehistory(self, time_history = int()):
        self.points = self.df[self.df[self.df.columns[1]] != 0]
        self.prehistory = pd.DataFrame()
        for i in reversed(self.points.index):
            self.temparr = self.df.loc[:i].tail(time_history)[self.df.columns[0]].values
            self.indicator = self.points.loc[i, self.points.columns[1]]
            
            if self.indicator == 2: self.indicator = 0
            self.linedf = []
            self.linedf.append(self.indicator)
            for i in self.temparr: self.linedf.append(i)
            
            if len(self.temparr) == time_history:
                self.temparr = [self.linedf]
                self.tempdf = pd.DataFrame(self.temparr)
                self.prehistory = pd.concat([self.prehistory, self.tempdf])
        
        self.prehistory.reset_index(inplace=True)
        self.prehistory = self.prehistory.iloc[:,1:] 
        self.prehistory.rename(columns={0:'self.indicator'}, inplace=True)
        #print(self.prehistory)
        return self.prehistory
    
    def zigzag(self, h):
        self.column_name = self.df.columns[0]
        self.indicators = peak_valley_pivots(self.df[self.column_name], h, -h)
        self.indicators *= -1
        
        self.firstidx = 0
        self.lastidx = 0
        for i in range(len(self.indicators)):
            if self.indicators[i] != 0:
                self.lastidx = self.firstidx
                self.firstidx = i
                if i != 0:
                    self.hoursdiff = self.firstidx-self.lastidx
                    self.coef = random.betavariate(alpha=2, beta=3)
                    self.idx_zeromark = round(self.coef*self.hoursdiff)+self.lastidx
                    self.indicators[self.idx_zeromark] = 2
            
        self.df.insert(1, "updown", self.indicators)
        # self.time_history = getDistribution(self.df, True)
        self.df_indicators = self.df[self.df[self.df.columns[1]] != 0]
        # print(f"Time history len: {self.time_history}self.h\nCount of self.indicator's point: {self.df_indicators.shape[0]}")
        # self.prehistory = getPrehistory(time_history=self.time_history)
        # os.makedirs('self.prehistory', exist_ok=True)
        # self.prehistory.to_csv(f'self.prehistory\dataset_{symbol}_{interval}_{self.df.columns[0]}.csv') #?Сохраняем в csv  
 
    
if __name__ == '__main__':
    pass