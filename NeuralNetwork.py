import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.autograd import Variable
import numpy as np
import random

INPUT_DIM = 8           #?Входные параметры
OUTPUT_DIM = 3          #?Выходные признаки 3 штуки т.к. (-1 0 1)
H_DIM1 = 6              #?Кол-во нейронов в первом слое
H_DIM2 = 6
H_DIM3 = 4
H_DIM4 = 4
LEARNING_RATE = 0.01 
EPOCHES = 1000

#df = pd.read_csv('D:\Programing\Project\prehistory\dataset_symbol=AAPL_interval=60min_close.csv')
df = pd.read_csv('prehistory\dataset_EURUSD_1m.csv')
df = df.iloc[:,1:]

print(df)
print(df.shape)
INPUT_DIM = df.shape[1]-1

dataset = []
for i in df.index.values:
    indicator = df.loc[i,df.columns[0]]
    data = df.loc[i, df.columns[1]:].values
    indicator_arr = []
    if indicator == -1:
        indicator_arr = [1, 0, 0]
    elif indicator == 0:
        indicator_arr = [0, 1, 0]
    elif indicator == 1:
        indicator_arr = [0, 0, 1]    
    dataset.append((indicator_arr, data.tolist()))
        
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.layer1 = nn.Linear(INPUT_DIM, H_DIM1)
        self.layer2 = nn.Linear(H_DIM1, H_DIM2)
        self.layer3 = nn.Linear(H_DIM2, H_DIM3)
        self.layer4 = nn.Linear(H_DIM3, H_DIM4)
        self.layer5 = nn.Linear(H_DIM4, OUTPUT_DIM)
    
    def forward(self, x):
        x = nnF.relu(self.layer1(x))
        x = nnF.relu(self.layer2(x))
        x = nnF.relu(self.layer3(x))
        x = nnF.relu(self.layer4(x))
        x = self.layer5(x)
        return nnF.softmax(input=x)

network = Network()
print(network)

optimizer = torch.optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHES):
    random.shuffle(dataset)
    for batch_idx, (target, data) in enumerate(dataset):
        
        data = torch.FloatTensor(data=data)
        target = torch.FloatTensor(data=target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        network_out = network(data)
        loss = criterion(network_out, target)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
            print(f'Train Epoch:{epoch}, \tLoss:{loss.data}')

#? Тестируем работу сети

for batch_idx, (target, data) in enumerate(dataset):
    data = torch.FloatTensor(data=data)
    target = torch.FloatTensor(data=target)
    data = Variable(data)
    target = Variable(target)
    network_out = network(data)
    print(f"Test number:{batch_idx}, target:{target}, out:{network_out.data}")    