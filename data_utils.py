import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap

def lin_schedule(t, a_0, a_1):
    #simple linear schedule on [0,1] s.t. f(0) = x_0 and f(1) = x_1
    return a_0*(1-t) + a_1*t

def lin_schedule_dt(t, a_0, a_1):
    return a_1 - a_0

def linear_traj(t, x_0, x_1):
    alpha_t = lin_schedule(t, 0, 1)
    sigma_t = lin_schedule(t, 1, 0)

    return alpha_t*x_1 + sigma_t*x_0


def linear_traj_dt(t, x_0, x_1):
    alpha_dt = lin_schedule_dt(t, 0, 1)
    sigma_dt = lin_schedule_dt(t, 1, 0)

    return alpha_dt*x_1 + sigma_dt*x_0


def generate_circle_flow(interpolant, batch_size=1, embedding=None, use_t=True):
    t = torch.rand(size=(batch_size,1))
    #x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
    x0 = torch.randn(size=(batch_size,2))*0.2
    x1 = torch.rand(size=(batch_size,2))*2.0 - 1.0
    #x1 = torch.randn(size=(batch_size,2))
    #x1 = x1/torch.norm(x1,dim=-1, keepdim=True)
    zero = torch.zeros(size=(batch_size,1))
    if embedding:
        x0_t = torch.cat((x0,zero),1)
        x0 = vmap(embedding)(x0_t)
        x0 = x0[:,:-1]

    X, Y = interpolant(x0,x1,t)

    if use_t:
        X = torch.cat((X,t), 1)

    return X, Y

def generate_circle_flow_DT(interpolant, batch_size=1, time_steps=10):
    #t_index = torch.randint(low=0,high=time_steps-1,size=(1,))
    #t = torch.linspace(0,1-1/time_steps, time_steps)[t_index] + 1/time_steps/2
    t_ = torch.rand(size=(1,))
    t = t_.repeat(batch_size, 1)
    #r = torch.rand(size=(batch_size, 1))**(1/2)
    #x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
    x0 = torch.randn(size=(batch_size,2))
    #x1 = torch.rand(size=(batch_size,2))*2.0 - 1.0
    x1 = torch.randn(size=(batch_size,2))
    x1 = x1/torch.norm(x1,dim=-1, keepdim=True)

    X, Y = interpolant(x0,x1,t)
    Y += torch.randn(size=(batch_size,2))*1e-3
    return X, Y, t_[0]

# def generate_circle_flow_DT(interpolant, batch_size=1, time_steps=10):
#   #t = torch.linspace(0,1,time_steps)
#   #t = t[None,:,None].repeat(1,batch_size,1)
#   x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
#   x1 = torch.randn(size=(batch_size,2))
#   x1 = x1/torch.norm(x1,dim=-1, keepdim=True)
#   Y = interpolant(x0,x1,time_steps)

#   return x0, Y


def generate_circle_flow_map(batch_size=1):
    x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
    x1 = torch.randn(size=(batch_size,2))
    x1 = x1/torch.norm(x1,dim=-1, keepdim=True)

    return x0, x1

def generate_grid_flow(interpolant, batch_size=1, scale=1.0):
    t = torch.rand(size=(batch_size,1))
    x0 = torch.randn(size=(batch_size,2))
    x1 = np.zeros((batch_size, 2))
    # iterative sampling for now (for my sanity)
    for i in range(batch_size):     
        #x_0 = np.random.normal(0,1,size=(2,))*0.2
        #r_0 = np.random.normal(0,1)**(1/2.)
        #x_0 = x_0 / np.linalg.norm(x_0)
        grid_range = [(0.,0.),(0,2./5),(0, 4./5),
                        (1./5, 1./5), (1./5, 3./5),
                        (2./5,0.),(2./5,2./5),(2./5, 4./5),
                        (3./5, 1./5), (3./5, 3./5),
                        (4./5,0.),(4./5, 2./5),(4./5, 4./5)]
        j = np.random.randint(0,13)
        x_11 = np.random.uniform(grid_range[j][1], grid_range[j][1]+1./5)
        x_12 = np.random.uniform(grid_range[j][0], grid_range[j][0]+1./5)
        x1[i,:] = np.array([scale*(x_11-0.5),scale*(x_12-0.5)])

    x1 = torch.from_numpy(x1).float()
    X, Y = interpolant(x0, x1, t)
    X = torch.cat((X, t), 1)

    return X, Y






