import os,sys,numpy as np

import torch

import time

def forloopdists(feats,protos):
  #shapes: are (N,D), (P,D)
  rows = feats.shape[0]
  cols = protos.shape[0]
  result = torch.zeros((rows, cols))
  #YOUR implementation here
  for i in range(rows):
    for j in range(cols):
      result[i, j] = torch.linalg.norm(torch.from_numpy(feats[i, :]) - torch.from_numpy(protos[j, :]))**2
  return result

def numpydists(feats,protos : np.ndarray):
  #shapes: are (N,D), (P,D)
  # Uses trick: ||A - B||^2 = ||A_i||^2 + ||B_j||^2.T - 2*A*B.T
  # Shapes:                   (5000, 1) + (1, 500) - (5000, 500) -> (5000, 500), broadcasting works
  return np.sum(feats**2, axis=1, keepdims=True) + np.sum(protos**2, axis=1, keepdims=True).T - 2*np.matmul(feats, protos.T)
  
  # else if appending 0th dim to N, D --> (1, N, D) and to 1th in P, D --> (P, 1, D) --> frankenstein of (P, N, D)
  
def pytorchdists(feats0,protos0,device):
  #shapes: are (N,D), (P,D)

  #YOUR implementation here
  return torch.sum(feats0**2, dim=1, keepdim=True) + torch.sum(protos0**2, dim=1, keepdim=True).t() - 2*torch.matmul(feats0, protos0.t())
  

def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  feats=np.random.normal(size=(250000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))
  feats_tensor = torch.from_numpy(feats)
  protos_tensor = torch.from_numpy(protos)


  '''
  since = time.time()
  dists0=forloopdists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  '''

  device=torch.device('cpu')
  since = time.time()
  
  
  # First with for loop
  
  # dists0 = forloopdists(feats, protos)
  # print(dists0)
  # print(dists0.shape)
  
  # print("For-loop completed in ", (time.time() - float(since)), "s")

  dists1=pytorchdists(feats_tensor,protos_tensor,device)
  print(dists1)
  print(dists1.shape)

  
  time_elapsed=float(time.time()) - float(since)

  print('PyTorch complete in {:.3f}s'.format( time_elapsed )) # PyTorch complete in 1.311s
  print(dists1.shape)

  # print('df0',np.max(np.abs(dists1-dists0)))
  since = time.time()

  dists2=numpydists(feats,protos)

  time_elapsed=float(time.time()) - float(since)

  print('Numpy complete in {:.3f}s'.format( time_elapsed )) # Numpy complete in 1.651s

  print(dists2.shape)

  # print('df',np.max(np.abs(dists1-dists2)))
  


if __name__=='__main__':
  run()
