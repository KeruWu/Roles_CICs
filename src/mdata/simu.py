"""Generate simulated data from Linear SCM
"""
import numpy as np
import torch
import itertools


def _create_dataloaders(data, M, batch_size = 100):
    """create torch dataloaders from list of (X, Y) pairs

    Args:
      data (list of (X, Y)): list of envs
      M (int): number of envs
      batch_size (int, optional): Defaults to 100.
    """
    dataloaders = []

    for i in range(M):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(data[i][0]),
            torch.Tensor(data[i][1]).long()
        )
        dataloaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
      
    return dataloaders
  

def linSCM_classic(M=2, n=500, d=9, batch_size=100):
    """generate data according a similar linear SCM
    mixture of two Gaussians

    Args:
      M (int, optional): number of envs.
      n (int, optional): sample size.
      d (int, optional): X dimension.
    """
    data = []
    yprobs = 0.5*np.ones(M)
    b = 0.2 * np.ones(d)

    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = Y.reshape(-1,1).dot(b.reshape(1, -1)) + 0.25*np.random.randn(n, d)
        data.append((X, Y))
        
    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


def SCM_1(M=10, n=500, d=9, batch_size=100, target=-1):
    """generate data according a similar linear SCM
    mixture of two Gaussians
    Domain adaptation setting 00:
    X shift, no Y shift, no mechanism shift

    Args:
      M (int, optional): number of envs. 
      n (int, optional): sample size.
      d (int, optional): X dimension.
    """
    data = []
    interventions = 0.2 * np.random.randn(M, d)
    # target X intervention is large
    interventions[target, :] = 1 * np.ones(d) * np.sign(np.random.randn(d))
    yprobs = 0.5*np.ones(M)
    b = 0.2 * np.ones(d)

    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = Y.reshape(-1,1).dot(b.reshape(1, -1)) + 0.25*np.random.randn(n, d) + interventions[m, :]
        data.append((X, Y))
        
    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


def SCM_2(M=10, n=500, d=9, batch_size=100, target=-1):
    """generate data according a similar linear SCM
    mixture of two Gaussians
    Domain adaptation setting 01:
    X shift, Y shift, no mechanism shift
    conditional invariant components CICs are present

    Args:
      M (int, optional): number of envs. 
      n (int, optional): sample size.
      d (int, optional): X dimension.
    """
    data = []
    interventions = 1 * np.random.randn(M, d)
    # target X intervention is large
    interventions[target, :] = 2 * np.ones(d) * np.sign(np.random.randn(d))
    # the last three coordinates are conditionally invariant
    interventions[:, -3:] = 0
    yprobs = 0.5*np.ones(M)
    # y label distribution shift for target env
    yprobs[target] = 0.1
    b = 0.2 * np.ones(d)


    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = (Y-0.5).reshape(-1,1).dot(b.reshape(1, -1)) + 0.25*np.random.randn(n, d) + interventions[m, :]
        data.append((X, Y))
        
    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


def SCM_3(M=12, n=500, d=18, batch_size = 100, target=-1):
    """generate data according a similar linear SCM
    mixture of two Gaussians
    Domain adaptation setting 03:
    X shift, no Y shift, mechanism shift
    conditional invariant components CICs are present

    Args:
      M (int, optional): number of envs.
      n (int, optional): sample size. 
      d (int, optional): X dimension.
    """
    data = []
    interventions = np.random.randn(M, d)
    # target X intervention is large
    interventions[target, :] = 2*np.ones(d) * np.sign(np.random.randn(d))
    # first 6 coordinates in the target domain is close to that of the first source domain
    base_intervention = 0.8 * np.random.randn(6)
    interventions[0, :6] = base_intervention + 0.6 * np.random.randn(6)
    interventions[target, :6] = base_intervention + 0.6 * np.random.randn(6)
    # the last three coordinates are conditionally invariant
    interventions[:, -6:] = 0
  
    # these indexes are going to have mechanism shift
    # the marginal distribution of X is unchanged
    # but the conditional distribution is completely wrong
    mech_envs = [0, 1, 2, 3, 4, 5] if M == 12 else [0]
    mech_ind_s = 6
    mech_ind_e = 12
    yprobs = 0.5*np.ones(M)
    b = 0.3 * np.ones(d)


    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = (Y-0.5).reshape(-1,1).dot(b.reshape(1, -1)) + 0.4*np.random.randn(n, d) + interventions[m, :]
        if m in mech_envs:
            X[:, mech_ind_s:mech_ind_e] = (0.5-Y).reshape(-1,1).dot(b[mech_ind_s:mech_ind_e].reshape(1, -1)) + 0.1*np.random.randn(n, mech_ind_e-mech_ind_s)
        else:
            X[:, mech_ind_s:mech_ind_e] = (Y-0.5).reshape(-1,1).dot(b[mech_ind_s:mech_ind_e].reshape(1, -1)) + 0.1*np.random.randn(n, mech_ind_e-mech_ind_s)
      
        data.append((X, Y))
        

    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


def SCM_4(M=12, n=500, d=9, batch_size=100, target=-1):
    """generate data according a similar linear SCM
    mixture of two Gaussians
    Domain adaptation setting 04:
    X shift, Y shift, mechanism shift
    conditional invariant components CICs are present

    Args:
      M (int, optional): number of envs.
      n (int, optional): sample size.
      d (int, optional): X dimension.
    """
    data = []
    interventions = 1 * np.random.randn(M, d)
    # target X intervention is large
    interventions[target, :] = 2 * np.ones(d) * np.sign(np.random.randn(d))
    # first 6 coordinates in the target domain is close to that of the first source domain
    base_intervention = 0.9 * np.random.randn(6)
    interventions[0, :6] = base_intervention + 0.44 * np.random.randn(6)
    interventions[target, :6] = base_intervention + 0.44 * np.random.randn(6)
    # the last three coordinates are conditionally invariant
    interventions[:, -6:] = 0

    # these indexes are going to have mechanism shift
    # the marginal distribution of X is unchanged
    # but the conditional distribution is completely wrong
    mech_envs = [0, 1, 2, 3, 4, 5] if M == 12 else [0]
    mech_ind_s = 6
    mech_ind_e = 12
    yprobs = 0.5*np.ones(M)
    yprobs[target] = 0.3
    b =  0.3* np.ones(d)

    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = (Y-0.5).reshape(-1,1).dot(b.reshape(1, -1)) + 0.4*np.random.randn(n, d) + interventions[m, :]
        if m in mech_envs:
            X[:, mech_ind_s:mech_ind_e] = (0.5-Y).reshape(-1,1).dot(b[mech_ind_s:mech_ind_e].reshape(1, -1)) + 0.1*np.random.randn(n, mech_ind_e-mech_ind_s)
        else:
            X[:, mech_ind_s:mech_ind_e] = (Y-0.5).reshape(-1,1).dot(b[mech_ind_s:mech_ind_e].reshape(1, -1)) + 0.1*np.random.randn(n, mech_ind_e-mech_ind_s)
      
        data.append((X, Y))
        

    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


def SCM_binary(M=10, n=500, d=10, batch_size = 100, target=-1):
    data = []
    interventions = np.zeros((M, d))
    A = np.array([p for p in itertools.product(*[[-1.,1.] for i in range(d//2)])])
    idx = np.random.choice(np.arange(2**(d//2)), M-1, replace=False)
    interventions[:(M-1),(d//2):] = A[idx]
    interventions[-1,(d//2):] = np.array([2]*(d//2))

    yprobs = 0.5*np.ones(M)
    b = 0.3 * np.ones(d)


    for m in range(M):
        Y = np.array(np.random.rand(n) < yprobs[m], dtype=int)
        X = (Y-0.5).reshape(-1,1).dot(b.reshape(1, -1)) + 0.4*np.random.randn(n, d) + interventions[m, :]
        data.append((X, Y))
        

    dataloaders = _create_dataloaders(data, M, batch_size)

    return dataloaders


