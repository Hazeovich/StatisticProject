import os
from readData import ReadData
from datasetPreparetion import Dataset

if __name__ == '__main__':
    data = ReadData('data/EURUSD_MIN.csv')
    data.read_csv_to_pd()
    raw_data = data.get_close_values()
    
    print(raw_data)
    
    dataset = Dataset(raw_data, 100)
    dataset.zigzag(0.01)
    dataset.showPlot()
    time_history = dataset.getDistribution(True)
    prehistory = dataset.getPrehistory(time_history=time_history)
    os.makedirs('prehistory', exist_ok=True)
    prehistory.to_csv('prehistory\dataset_EURUSD_1m.csv') #?Сохраняем в csv 