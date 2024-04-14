import os

os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file = os.path.join('./', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')


# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 插值法填充数据，也可以考虑删除法
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
print(type(inputs))

import torch

# print(torch.tensor(inputs)) #出错，pandas dataframe不能直接转换为tensor

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)


# practice
print('''-----Practice-----''')

os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file = os.path.join('./', 'data', 'practice.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Add,Price\n')  # 列名
    f.write('NA,Pave,1544,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,45,106000\n')
    f.write('4,NA,13254,178100\n')
    f.write('NA,NA,154242,140000\n')
    f.write('2,NA,NA,106000\n')
    f.write('4,NA,NA,178100\n')
    f.write('NA,NA,11,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:3], data.iloc[:, 3]

# practice1
missing_counts = inputs.isnull().sum()  # 统计每列的缺失值数量
print(missing_counts)
column_with_max_missing = missing_counts.idxmax()  # 找到缺失值最多的列名
print(column_with_max_missing)
inputs = inputs.drop(column_with_max_missing, axis=1)  # 删除缺失值最多的列
print(inputs)

# practice2
inputs = inputs.fillna(inputs.mean())
inputs = pd.get_dummies(inputs, dummy_na=True)
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)