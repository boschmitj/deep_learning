import torch
import numpy as np

seed = 42
rng = torch.manual_seed(42)

def rand_t(dim0=1, dim1=1):
    return torch.randn(dim0, dim1, generator=rng).detach_().requires_grad_()

def rand_v(dim):
    return torch.randn(dim).detach_().requires_grad_()

def rand_s():
    return torch.randn((), generator=rng).detach_().requires_grad_()

def check_scalar(t):
    if not t.ndim == 0:
        print("Resulting tensor is not scalar!")
        exit()
    

def f1(x, a, b, d):
    if x.item() <= 0:
        print("x > 0 not fulfilled")
        exit()
    return a*(x**6) + b*(x**4) + d * torch.log(x)

def f2(x):
    return torch.e**(-x)*torch.cos(-2*x)

class f3:
    def __init__(self, b, x, w):
        self.g = lambda z: 1 / (1 + torch.e**(-z))
        self.b = b
        self.x = x
        self.w = w
        
    def forward(self):
        result = self.g(torch.dot(self.w, self.x) + self.b)
        check_scalar(result)
        return result 
    
class f4:
    def __init__(self, a, v, X):
        self.a = a
        self.v = v
        self.X = X
        
    def forward(self):
        Xa = torch.mm(self.X, self.a)
        result = torch.mm(self.v, Xa).sum() # konvertiert [1, 1] tensor zu [] tensor (=scalar)
        check_scalar(result)
        return result
    
class f5:
    def __init__(self, v, z, B, A, X):
        self.v = v
        self.z = z
        self.B = B
        self.A = A
        self.X = X
    
    def forward(self):
        XXt = torch.mm(self.X, torch.t(self.X))
        Bz = torch.mm(self.B, self.z)
        XX_tBz = torch.mm(XXt, Bz)
        AXXBz = torch.mm(self.A, XX_tBz)
        result = torch.mm(self.v, AXXBz)
        print(result)
        print(result.shape)
        result = result.sum()
        print(result)
        check_scalar(result)
        return result
            
class Computer:
        
    def computef1(self):
        print("\n")
        print("Computing f1")
        a, b, d = rand_s(), rand_s(), rand_s()
        x = rand_s()
        f = f1(x, a, b, d)
        print("a:", a, "b:", b, "d:", d, "x:", x)
        grads = torch.autograd.grad(f, [x, a, b, d])
        print("Gradient:", grads)
        
    def computef2(self):
        print("\n")
        print("Computing f2")
        x = rand_s()
        f = f2(x)
        print("x is", x)
        grads = torch.autograd.grad(f, [x])
        print("Gradient:", grads)
    
    def computef3(self, d):
        print("\n")
        print("Computing f3")
        b = rand_s()
        x = rand_v(d)
        w = rand_v(d)
        print("b", b, "x", x, "w", w)
        f3c = f3(b, x, w)
        loss = f3c.forward()
        print(f"f({x})={loss}")
        grads = torch.autograd.grad(loss, [b, x, w], allow_unused=True)
        print("Gradients", grads)
        
        
    def computef4(self, k=2, d=3):
        print("\n")
        print("Computing f4")
        a = rand_t(k, 1)
        print(a.shape)
        v = rand_t(1, d)
        print(v.shape)
        X = rand_t(d, k)
        print(X.shape)
        print("a", a, "v", v, "X", X)
        
        f4c = f4(a, v, X)
        result = f4c.forward()
        print(f"f(X)={result}")
        
        grads = torch.autograd.grad(result, [a, v, X])
        print("Gradients", grads)
        
    def computef5(self, n=2, d=3):
        print("\n")
        print("Computing f5")
        
        v = rand_t(dim0=1, dim1=d)
        z = rand_t(dim0=d, dim1=1)
        B = rand_t(dim0=d, dim1=d)
        A = rand_t(dim0=d, dim1=d)
        print("v", v, "z", z, "B", B, "A", A)
        
        X = rand_t(d, n)
        
        f5c = f5(v, z, B, A, X)
        result = f5c.forward()
        grads = torch.autograd.grad(result, [v, z, B, A, X])
        
        print("Result:", result)
        print("Gradients", grads)
        
    
    
if __name__=="__main__":
    computer = Computer()
    computer.computef1()
    computer.computef2()
    computer.computef3(3)
    computer.computef4(2, 3)
    computer.computef5(2, 3)
    