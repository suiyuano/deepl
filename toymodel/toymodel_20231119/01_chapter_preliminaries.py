import pandas
import torch
import os
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #生成数据
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Wc,Price\n')  # 列名
        f.write('NA,Pave,3,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,2,106000\n')
        f.write('4,NA,NA,178100\n')
        f.write('NA,NA,1,140000\n')

    data = pd.read_csv(data_file)
    print(data)

    # 统计缺失最多的列并删除
    miss = pandas.isna(data).sum()
    print(miss)
    maxkey = miss.idxmax()
    print(maxkey)
    data = data.drop(maxkey, axis=1)
    print(data)

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # 先把字符串类型的缺失值补全
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)
    # 再把数字类型的缺失补全
    inputs = inputs.fillna(inputs.mean())
    print(inputs)
    X = torch.tensor(inputs.to_numpy(dtype=float))
    Y = torch.tensor(outputs.to_numpy(dtype=float))
    print(X, Y)

