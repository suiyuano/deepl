import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = torch.arange(12)
    print(x)
    print(x.shape)
    X = x.reshape(3, 4)
    print(X)
    print(torch.zeros((2, 3, 4)))

    # test
    a = torch.arange(12).reshape(2, 3, -1)
    b = torch.arange(12).reshape(2, 3, -1)
    print(a == b)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
