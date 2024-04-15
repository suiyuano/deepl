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