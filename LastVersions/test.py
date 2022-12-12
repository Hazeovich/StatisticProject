import pandas as pd

arr=[
        [-1, [10,20,30,40,50,60,70,80,90,100]],
        [0, [10,20,30]], 
        [1, [10,20,30]]
]

df = pd.DataFrame(arr)
print(df)




# frames = [1,2,3,4]
# dates = ['2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04']
# values = [10,15,20,25, 30,35,40,45, 50,55,60,65, 70,75,80,85]




# arrays = [[1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4],
#           [1,1,1,1, -1,-1,-1,-1, 1,1,1,1, -1,-1,-1,-1, ], 
#           ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
#            '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', 
#            '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', 
#            '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04']]

# index = pd.MultiIndex.from_arrays(arrays, names=('frame','updown', 'date'))
# data = {'value':values}

# arr2 = [[5,5,5,5],
#         [1,1,1,1],
#         dates]
# index2 = pd.MultiIndex.from_arrays(arr2, names=('frame','updown', 'date'))
# data2 = {'value':[90,95,100,105]}

# df = pd.DataFrame(data=data, index=index)
# df2 = pd.DataFrame(data=data2, index=index2)
# print(df)
# print(df2)
# df = pd.concat([df, df2])
# print(df[(df.value != 10)])

# arr3 = [[],[],[]]

# arr3[0].append(1)
# print(arr3)
