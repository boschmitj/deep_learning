import torch
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, inNum, outNum):
        super().__init__()
        self.Dense = nn.Sequential(
            nn.Linear(inNum, outNum),
            nn.ReLU()
        )
        
    def forward(self, X):
        return self.Dense(X)
    
class Dense2(nn.Module):
    def __init__(self, inNum, outNum):
        super().__init__()
        self.lin1 = nn.Linear(inNum, outNum)
        self.lin1.weight = nn.Parameter(torch.randn(inNum, outNum))
        self.lin1.bias = nn.Parameter(torch.ones(inNum, outNum))
        
        
    def forward(self, X):
        out = self.lin1(X)
        out = torch.relu(out)
        return out