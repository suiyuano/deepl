# import matplotlib
import pandas
import torch
import os
import pandas as pd
# from jedi.api.refactoring import inline
# %matplotlib inline
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
from matplotlib import pyplot as plt


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


def pandas_learn():
    # 生成数据
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


def calculus_learn():
    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
        h *= 0.1

    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


def autograd():
    x = torch.arange(4.0)
    print(x)
    x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
    print(x.grad)  # 默认值是None
    y = 2 * torch.dot(x, x)
    print(y)
    y.backward()
    print(x.grad)
    print(x.grad == 4 * x)

    # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(x.grad)

    # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
    # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
    x.grad.zero_()
    y = x * x
    # 等价于y.backward(torch.ones(len(x)))
    y.sum().backward()
    print(x.grad)


def main():
    # pandas_learn()
    # calculus_learn()
    autograd()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
