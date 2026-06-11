import torch

def f(x, y):
    return torch.sin((x**2)*y)

(x, y) = (torch.tensor(1.2, requires_grad=True), torch.tensor(3.4, requires_grad=True))
out = f(x, y)
print(out)
out.backward()
print(x.grad, y.grad)
