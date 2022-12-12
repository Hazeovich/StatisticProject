import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

INPUT_DIM = 10          #?Входные параметры
OUTPUT_DIM = 3          #?Выходные признаки 3 штуки т.к. (-1 0 1)
H_DIM = 15              #?Кол-во нейронов в первом слое
LEARNING_RATE = 0.0002  #?Шаг градиентного спука или скорость обучения
EPOCHS = 1000           #?Количество эпох
BATCH_SIZE = 10          #?Размер батча

#? Достаем данные из датасета
dataset = pd.read_csv('prehistory/out_AAPL.csv')

tmp_val = dataset.iloc[:,3].values
tmp_arr = [dataset.get('frame').values,dataset.get('updown').values,dataset.get('date').values]
tmp_index = pd.MultiIndex.from_arrays(tmp_arr, names=('frame','updown', 'date'))  
dataset = pd.DataFrame(data=tmp_val, index=tmp_index, columns=['AAPL'])
frame_counter = list(set(dataset.index.get_level_values('frame').values))

dataset_train = []
for i in frame_counter:
    output_val = list(set(dataset.loc[i].index.get_level_values('updown').values))[0]
    input_arr = np.array(dataset.loc[i].values).ravel().tolist()
    if len(input_arr) == 10: 
        if output_val == -1:
            dataset_train.append((input_arr, 0)) #На убль
        elif output_val == 0:
            dataset_train.append((input_arr, 1)) #Нейтрально
        elif output_val == 1:
            dataset_train.append((input_arr, 2)) #На рост

#? Создаем нейронку на NumPy
W1 = np.random.randn(INPUT_DIM, H_DIM)      #Матрица весов с размерами: 10 строк и 5 столбцов 
B1 = np.random.randn(1, H_DIM)                 #Число коэфов b равно числу нейронов в полносвязном слое
W2 = np.random.randn(H_DIM, OUTPUT_DIM)     #Матрица весов с размерами: 5 строк и 3 столбцов 
B2 = np.random.randn(1, OUTPUT_DIM)            #Число коэфов b равно числу нейронов в следующем слое

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
B1 = (B1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
B2 = (B2 - 0.5) * 2 * np.sqrt(1/H_DIM)

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z, y):
    return -np.sum(y*np.log(z))

def sparse_cross_entropy_batch(z, y):
    return -np.log([z[j, y[j]] for j in range(len(y))])

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full  = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def relu_derivate(t):
    return (t >= 0).astype(float)

#?Forward

loss_arr = []
for ep in range(EPOCHS):
    random.shuffle(dataset_train)
    for i in range(len(dataset_train) // BATCH_SIZE):
        
        batch_x, batch_y = zip(*dataset_train[i*BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE])
        X = np.concatenate(batch_x, axis=0)
        X = X.reshape((BATCH_SIZE, INPUT_DIM))
        Y = np.array(batch_y, ndmin=2)
        # x, y = dataset_train[i]
        # X = np.array(x, ndmin=2)
        # Y = np.array(y, ndmin=2)

        T1 = X @ W1 + B1
        Y1 = relu(T1)
        T2 = Y1 @ W2 + B2
        Y2 = softmax_batch(T2)
        Err = np.sum(sparse_cross_entropy_batch(Y2, Y))

        #?Backward
        y_full = to_full_batch(Y, OUTPUT_DIM)
        dErr_dT2 = Y2 - y_full
        dErr_dW2 = Y1.T @ dErr_dT2
        dErr_dB2 = np.sum(dErr_dT2, axis=0, keepdims=True)
        dErr_dY1 = dErr_dT2 @ W2.T
        dErr_dT1 = dErr_dY1 * relu_derivate(T1)
        dErr_dW1 = X.T @ dErr_dT1
        dErr_dB1 = np.sum(dErr_dT1, axis=0, keepdims=True)
        
        # y_full = to_full(Y, OUTPUT_DIM)
        # dErr_dT2 = Y2 - y_full
        # dErr_dW2 = Y1.T @ dErr_dT2
        # dErr_dB2 = dErr_dT2
        # dErr_dY1 = dErr_dT2 @ W2.T
        # dErr_dT1 = dErr_dY1 * relu_derivate(T1)
        # dErr_dW1 = X.T @ dErr_dT1
        # dErr_dB1 = dErr_dT1

        #?Update
        W1 = W1 - LEARNING_RATE * dErr_dW1
        B1 = B1 - LEARNING_RATE * dErr_dB1
        W2 = W2 - LEARNING_RATE * dErr_dW2
        B2 = B2 - LEARNING_RATE * dErr_dB2
        
        loss_arr.append(Err)

def predict(X):
    T1 = X @ W1 + B1
    Y1 = relu(T1)
    T2 = Y1 @ W2 + B2
    Y2 = softmax(T2)
    return Y2

def calc_accuracy():
    correct = 0
    for x, y in dataset_train:
        X = np.array(x, ndmin=2)
        Y = np.array(y, ndmin=2)
        Z = predict(X)
        Y_pred = np.argmax(Z)
        if Y_pred == Y:
            correct += 1
        print(Z, Y)
    acc = correct / len(dataset_train)
    return acc
  
accuracy = calc_accuracy()
print("Accuracy:", accuracy)

plt.plot(loss_arr)
plt.show()