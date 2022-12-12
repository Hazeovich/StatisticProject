import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.autograd import Variable

INPUT_DIM = 10          #?Входные параметры
OUTPUT_DIM = 3          #?Выходные признаки 3 штуки т.к. (-1 0 1)
H_DIM1 = 6              #?Кол-во нейронов в первом слое
H_DIM2 = 25
H_DIM3 = 8
H_DIM4 = 6
LEARNING_RATE = 0.01 
EPOCHES = 1000

#? Достаем данные из датасета
dataset = pd.read_csv('prehistory/out_AAPL.csv')

tmp_val = dataset.iloc[:,3].values
tmp_arr = [dataset.get('frame').values,dataset.get('updown').values,dataset.get('date').values]
tmp_index = pd.MultiIndex.from_arrays(tmp_arr, names=('frame','updown', 'date'))  
dataset = pd.DataFrame(data=tmp_val, index=tmp_index, columns=['AAPL'])
frame_counter = list(set(dataset.index.get_level_values('frame').values))

input_list = []
output_list = []

for i in frame_counter:
    output_val = list(set(dataset.loc[i].index.get_level_values('updown').values))[0]
    input_arr = np.array(dataset.loc[i].values).ravel().tolist()
    if len(input_arr) == 10: 
        input_list.append(input_arr)
        if output_val == -1:
            output_list.append([1, 0, 0]) #На убль
        elif output_val == 0:
            output_list.append([0, 1, 0]) #Нейтрально
        elif output_val == 1:
            output_list.append([0, 0, 1]) #На рост

train_loader = []
for i in range(len(input_list)):
    train_loader.append((input_list[i], output_list[i]))


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

print(train_loader)

for epoch in range(EPOCHES):
    for batch_idx, (data, target) in enumerate(train_loader):
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

for batch_idx, (data, target) in enumerate(train_loader):
    data = torch.FloatTensor(data=data)
    target = torch.FloatTensor(data=target)
    data = Variable(data)
    target = Variable(target)
    network_out = network(data)
    print(f"Test number:{batch_idx}, target:{target}, out:{network_out.data}")    