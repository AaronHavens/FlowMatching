import torch
import torch.nn as nn


class simpleVF(nn.Module):
    def __init__(self, input_dim, hid_dim=128):
        super(simpleVF, self).__init__()
        # input includes stacked time dimension
        activation = nn.ReLU()
        self.net = nn.Sequential(   nn.Linear(input_dim+1, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, input_dim))

    def forward(self, x):
        return self.net(x)



class BezierInterpolant2(nn.Module):
    def __init__(self, state_dim, hid_dim=64):
        super(BezierInterpolant2, self).__init__()

        activation = nn.ReLU()
        self.state_dim = state_dim
        self.control_point = nn.Sequential(nn.Linear(state_dim*2, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, state_dim))

    def forward(self, x0, x2, t):
        z = torch.cat((x0,x2),1)
        x1 = self.control_point(z)
        bezier = x1 + (1-t)**2 * (x0 - x1) + t**2 * (x2-x1)
        dt_bezier = 2*(1-t)*(x1-x0) + 2*t*(x2-x1)

        return bezier, dt_bezier

class BezierInterpolant3(nn.Module):
    def __init__(self, state_dim, hid_dim=64):
        super(BezierInterpolant3, self).__init__()

        activation = nn.ReLU()
        self.state_dim = state_dim
        self.control_points = nn.Sequential(nn.Linear(state_dim*2, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, state_dim*2))

    def forward(self, x0, x3, t):
        z = torch.cat((x0,x3),1)
        x = self.control_points(z)
        x1 = x[:,:self.state_dim]
        x2 = x[:,self.state_dim:]
        bezier = (1-t)**3*x0 + 3*(1-t)**2 * t * x1 + 3*(1-t) * t**2 * x2 + t**3 * x3
        dt_bezier = 3*(1-t)**2 * (x1-x0) + 6*(1-t)*t*(x2-x1) + 3*t**2*(x3-x2)

        return bezier, dt_bezier


class LinearInterpolant(nn.Module):
    def __init__(self):
        super(LinearInterpolant, self).__init__()

    def forward(self, x0, x1, t):
        xt = (1-t)*x0 + t*x1
        dt_xt = x1 - x0
        return xt, dt_xt

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