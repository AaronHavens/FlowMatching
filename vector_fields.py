import torch
import torch.nn as nn


class simpleVF(nn.Module):
    def __init__(self, input_dim, hid_dim=128):
        super(simpleVF, self).__init__()
        # input includes stacked time dimension

        self.net = nn.Sequential(   nn.Linear(input_dim+1, hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(hid_dim, hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(hid_dim, hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(hid_dim, input_dim))

    def forward(self, x):
        return self.net(x)




def wall_barrier(x):
    x1 = x[:,0]

    return -torch.square(x1) + 0.5**2

def wall_barrier_cond(x, V):
    x1 = x[:,0]
    grad_h_x1 = -2*x1


    return grad_h_x1 * V[:,0] + 10.*wall_barrier(x)

def hole_barrier(x):
    x1 = x[:,0]
    x2 = x[:,1]

    return torch.square(x1) + torch.square(x2-0.5) - 0.25**2

def hole_barrier_cond(x, V):
    x1 = x[:,0]
    x2 = x[:,1]

    grad_h_x1 = 2*x1
    grad_h_x2 = 2*(x2-0.5)

    return grad_h_x1 * V[:,0] + grad_h_x2 * V[:,1] + 10. * hole_barrier(x) 