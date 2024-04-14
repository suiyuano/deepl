import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
X = x.reshape(3, 4)
print(X)
X_same = x.reshape(-1, 4)
print(X_same)
X_same_too = x.reshape(3, -1)
print(X_same_too)
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn(3, 4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

print(X, X[-1], X[1:3])
X[1, 2] = 9
print(X)

X[0:2, :] = 12
print(X)

before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

# practice
print('''-----Practice-----''')

# practice1
print(X, Y)
print(X < Y)

# practice2
m = torch.arange(12).reshape((2, 2, 3))
n = torch.arange(12).reshape((2, 2, 3))
print(m, n)
print(m+n)
