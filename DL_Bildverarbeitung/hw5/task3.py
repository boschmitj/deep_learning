import torch
import task2

def xtAx(x, A):
    if not x.shape[0] == A.shape[0] == A.shape[1]:
        print("Wrong shapes")
        exit()
    y = torch.t(x) @ A @ x
    return y.squeeze()

def f(x):
    return 2*(x**3) - 5*(x**2) + 3*x - 7

class Computer:
    def __init__(self, n):
        self.n = n
        
        
    def verify(self, deriv1, x, A):
        a = deriv1
        b = (x.unsqueeze(0) @ (A + A.T)).squeeze()
        err = torch.norm(a - b).item()
        ok = err < 1e-6
        if not ok:
            print(f"[verify] first derivative mismatch, error={err:.3e}")
            print("autograd:", a)
            print("expected:", b)
        return ok
    
    def verifySecond(self, deriv2, w, A):
        a = deriv2
        b = (w.unsqueeze(0) @ (A + A.T)).squeeze()
        err = torch.norm(a - b).item()
        ok = err < 1e-6
        if not ok:
            print(f"[verify] second derivative mismatch, error={err:.3e}")
            print("autograd:", a)
            print("expected:", b)
        return ok
    
    def compute(self):
        n = self.n
        print(f"[compute] n={n}")
        x = task2.rand_v(n)
        A = task2.rand_t(n, n).requires_grad_(False)
        y = xtAx(x, A)
        print(f"[compute] y = x^T A x: {y.item():.6f}")
        deriv1 = torch.autograd.grad(y, x, create_graph=True)[0]
        print("[compute] first derivative d(x^T A x)/dx:", deriv1)
        
    
        if not self.verify(deriv1, x, A):
            print("[result] first derivative check failed")
            exit()
        print("[result] first derivative check passed")

        w = self.one_hot_vector(n, i = 0)
        print("[compute] using one-hot direction i=0")
        
        y2 = deriv1.unsqueeze(0) @ w.unsqueeze(1)
        print(f"[compute] projected first derivative: {y2.item():.6f}")
        
        deriv2 = torch.autograd.grad(y2, x)[0]
        print("[compute] second derivative (directional):", deriv2)
        
        if not self.verifySecond(deriv2, w, A):
            print("[result] second derivative check failed")
            exit()
        print("[result] second derivative check passed")
            
    
    def compute2(self):
        x = task2.rand_s()
        y = f(x)
        print(f"Computed f(x): f({x})={y}")

        deriv1 = torch.autograd.grad(y, x, create_graph=True)
        print("First derivative", deriv1)
        
        deriv2 = torch.autograd.grad(deriv1, x)
        print("Second derivative", deriv2)
    
    def one_hot_vector(self, n, i):
        if i > n:
            print("i > n, error")
            exit()
        return torch.concat((torch.zeros(i), torch.ones(1), torch.zeros(n - i - 1)))
    
        
if __name__=="__main__":
    comp = Computer(5)
    comp.compute()
    
    print("For f(x)")
    comp.compute2()
    