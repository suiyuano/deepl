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