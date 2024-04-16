import torch

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
print(y)

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
print(y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
# y.backward()  #这个会报错
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)

# practice
print('''-----Practice-----''')

# practice2
x.grad.zero_()
y = x * x
print(y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
# y.backward()  #这个会报错
print(x.grad)
# y.sum().backward()  #马上再次求导会报错

# practice5
import torch

# x.grad.zero_()

# x = torch.rand(2, 4, requires_grad = True)
x = torch.linspace(0, 2 * torch.pi, 100,  requires_grad = True)  # 生成100
y = torch.sin(x)
y.sum().backward()

import numpy as np
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline

plt.figure(figsize=(8, 8))

plt.plot(x.detach().numpy(), y.detach().numpy(), "-", label="sin(x)")
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), "r--", label="cos(x)")

# x1 = torch.linspace(0, 2 * torch.pi, 200)
# plt.plot(x1.numpy(), torch.zeros_like(x1).numpy(), "--", label="0")

plt.legend(loc='upper left')

# 添加标签和标题
plt.xlabel('X')
plt.ylabel('f(x)')

# 显示图表

plt.show()