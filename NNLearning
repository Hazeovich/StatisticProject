import numpy as np
import pandas as pd
import torch

INPUT_DIM = 10          #?Входные параметры
OUTPUT_DIM = 3          #?Выходные признаки 3 штуки т.к. (-1 0 1)
H_DIM = 5               #?Кол-во нейронов в первом слое


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
            output_list.append([1, 0, 0])
        elif output_val == 0:
            output_list.append([0, 1, 0])
        elif output_val == 1:
            output_list.append([0, 0, 1])

#? Создаем нейронку на NumPy

W1 = torch.tensor(np.random.randn(INPUT_DIM, H_DIM), requires_grad=True)     #Матрица весов с размерами: 10 строк и 5 столбцов 
B1 = torch.tensor(np.random.randn(H_DIM), requires_grad=True)                 #Число коэфов b равно числу нейронов в полносвязном слое
W2 = torch.tensor(np.random.randn(H_DIM, OUTPUT_DIM), requires_grad=True)     #Матрица весов с размерами: 5 строк и 3 столбцов 
B2 = torch.tensor(np.random.randn(OUTPUT_DIM), requires_grad=True)            #Число коэфов b равно числу нейронов в следующем слое

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

W1 = W1.to(device=device)
B1 = B1.to(device=device)
W2 = W2.to(device=device)
B2 = B2.to(device=device)

def relu(t):
    return torch.maximum(t, torch.tensor(0))

def softmax(t):
    out = torch.exp(t)
    return out / torch.sum(out)
    
def predict(X):
    T1 = X @ W1 + B1
    Y1 = relu(T1)
    T2 = Y1 @ W2 + B2
    Y2 = softmax(T2)
    return Y2

def lossFunction(true_ans, predict_ans):
    return -torch.sum(true_ans * torch.log(predict_ans))

X = torch.tensor(input_list[0], dtype=torch.float64)
X = X.to(device=device)
Y = torch.tensor(output_list[0])
Y = Y.to(device=device)

step = 0.001

def make_gradient_step():
    Z = predict(X)
    Z = Z.to(device)
    Error = lossFunction(Y, Z)
    
    W1.retain_grad()
    B1.retain_grad()
    W2.retain_grad()
    B2.retain_grad()
    Error.backward()

    W1.data -= step * W1.grad
    B1.data -= step * B1.grad
    W2.data -= step * W2.grad
    B2.data -= step * B2.grad

    W1.grad.zero_()
    B1.grad.zero_()
    W2.grad.zero_()
    B2.grad.zero_()

for i in range(500):
    make_gradient_step()
    Z = predict(X)
    Z = Z.to(device)
    print(Z, Y)

print()
Z = predict(X)
Z = Z.to(device)
print(Z, Y)

newX = torch.tensor(input_list[3], dtype=torch.float64)
newX = newX.to(device=device)
newY = torch.tensor(output_list[3])
newY = newY.to(device=device)

newZ = predict(newX)
newZ = newZ.to(device)
print(newZ, newY)

# optimizer = torch.optim.SGD([W1, B1, W2, B2], lr=step)
# def make_gradient_step(input_data, output_data):
#     Z = predict(input_data)
#     Error = lossFunction(output_data, Z)
#     Z.retain_grad()
#     Error.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    
    

# for i in range(500):
#     make_gradient_step(X, Y)
#     print(f'iteration:{i}')

# newX = torch.tensor(input_list[1], dtype=torch.float64)
# newX = newX.to(device=device)
# newY = torch.tensor(output_list[1])
# newY = newY.to(device=device)
# print(predict(newX), newY)

# step = 0.001
# x = torch.tensor(W1)
# print(x)

# # x = torch.tensor([8.,  8.], 
# #                  requires_grad=True)

# optimizer = torch.optim.SGD([x], lr=step)

# def make_gradient_step(func, var):
#     func_res = func(var)
#     var.retain_grad()
#     func_res.backward()
#     optimizer.step()
#     optimizer.zero_grad()



# # for i in range(500):
# #     make_gradient_step(, x)
# #     print(x)