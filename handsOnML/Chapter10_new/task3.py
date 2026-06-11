from sklearn import datasets, model_selection
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from task2 import Dense
import optuna
import torchmetrics
from functools import partial
from sklearn.preprocessing import StandardScaler


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)



X, y = datasets.fetch_covtype(download_if_missing=True, return_X_y=True, shuffle=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Sample X", X.shape)

print("Sample y", y[:3])

class CovDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y - 1, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
train_dataset = CovDataset(X_train, y_train)
test_dataset = CovDataset(X_test, y_test)
val_dataset = CovDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, 128, True, num_workers=10, persistent_workers=True)
test_loader = DataLoader(test_dataset, 128, False, num_workers=10, persistent_workers=True)
val_loader = DataLoader(val_dataset, 128, False, num_workers=10, persistent_workers=True)

class MlpClf(nn.Module):
    def __init__(self, in_num, numcl, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            Dense(in_num, n_hidden),
            Dense(n_hidden, n_hidden),
            Dense(n_hidden, n_hidden),
            nn.Linear(n_hidden, numcl),
        )
        
    def forward(self, X):
        return self.net(X)
    
def objective(trial : optuna.Trial, train_loader, valid_loader):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = MlpClf(in_num=54, n_hidden=n_hidden, numcl=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )
    criterion = nn.CrossEntropyLoss()
    train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=7).to(device)
    val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=7).to(device)
    test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=7).to(device)
    train_and_eval_model(model, optimizer, criterion, 32, train_metric, val_metric, train_loader, valid_loader)
    test_acc = evaluate_model(model, test_metric, test_loader)
    print(f"Accuracy on test set: {test_acc}")
    return test_acc
    
def train_and_eval_model(model, optimizer, criterion, n_epoch, train_metric, val_metric,train_loader, val_loader):
    for epoch in range(n_epoch):
        print("Epoch", epoch)
        train_metric.reset()
        val_metric.reset()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):#
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)
            train_metric.update(y_pred, target)
            # print(f"Loss in batch {batch_idx} is {loss}")
            loss.backward()
            optimizer.step()
        
        print(f"Train Accuracy {train_metric.compute()}")
    
        # Eval loop
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            with torch.no_grad():
                y_pred = model(data)
                val_metric.update(y_pred, target)

        print(f"Val Accuracy {val_metric.compute()}")
    
def evaluate_model(model, metric, loader):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

objective_with_data = partial(objective, train_loader=train_loader, valid_loader=val_loader)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective_with_data, n_trials=5)

            
    

