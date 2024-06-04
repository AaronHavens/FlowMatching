import torch
import torch.nn as nn
from layers import *
from torch.func import jacrev, vmap
from torchdiffeq import odeint_adjoint as odeint

class SimpleVF(nn.Module):
    def __init__(self, input_dim, hid_dim=128):
        super(SimpleVF, self).__init__()
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



class AffineVF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        return self.W(x)

class SimpleAutVF(nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU, hid_dim=128):
        super().__init__()
        # input includes stacked time dimension

        self.net = nn.Sequential(   nn.Linear(input_dim, hid_dim),
                                    activation(),
                                    nn.Linear(hid_dim, hid_dim),
                                    activation(),
                                    nn.Linear(hid_dim, hid_dim),
                                    activation(),
                                    # nn.Linear(hid_dim, hid_dim),
                                    # activation(),
                                    # nn.Linear(hid_dim, hid_dim),
                                    # activation(),
                                    nn.Linear(hid_dim, input_dim))

    def forward(self, x):
        return self.net(x)

class DiscreteLinearVF(nn.Module):
    def __init__(self, input_dim, time_steps=10):
        super().__init__()
        # input includes stacked time dimension

        activation = nn.LeakyReLU()
        self.time_steps = time_steps
        self.dt = 1/time_steps
        self.vf_list = nn.ModuleList([AffineVF(input_dim) for j in range(time_steps)])

    def forward(self, x, t):
        k = int(t//self.dt)
        return self.vf_list[k](x)

    def integrate(self, x, t):
        # for j in reversed(range(self.time_steps)):
        #     x = self.layers[j](x) * self.dt
        return x + self.vf_list[t](x) * self.dt

class ODEIntModule(nn.Module):
    def __init__(self, vf):
        super().__init__()
        self.vf = vf

    def forward(self, t, x):
        return self.vf(x)
        

class DiscreteSimpleVF(nn.Module):
    def __init__(self, input_dim, time_steps=10):
        super().__init__()
        # input includes stacked time dimension

        activation = nn.ReLU
        self.time_steps = time_steps
        self.dt = 1/time_steps
        self.vf_list = nn.ModuleList([SimpleAutVF(input_dim, activation) for j in range(time_steps)])

    def forward(self, x, t):
        k = int(t//self.dt)
        return self.vf_list[k](x)

    def euler_integrate(self, x, t):
        # for j in reversed(range(self.time_steps)):
        #     x = self.layers[j](x) * self.dt
        return x + self.vf_list[t](x) * self.dt

    def midpoint_integrate(self, x, t):
        x_mid = x + self.vf_list[t](x) * self.dt / 2
        return x + self.vf_list[t](x_mid) * self.dt

    def odeint_integrate(self, x, t):
        #f_t = lambda tau, y : self.vf_list[t](y)
        f_t = ODEIntModule(self.vf_list[t])
        y_t = odeint(f_t, x, t=torch.Tensor([0, self.dt]))
        return y_t[-1]

class LipschitzVF(nn.Module):
    def __init__(self, input_dim, hid_dim=128):
        super(LipschitzVF, self).__init__()
        gamma = 4.0#(2.0)**(1/4)
        # input includes stacked time dimension
        activation = nn.ReLU()
        # self.net = nn.Sequential(   SDPLin(input_dim+1, hid_dim, gamma=gamma),
        #                             activation,
        #                             SDPLin(hid_dim, hid_dim, gamma=gamma),
        #                             activation,
        #                             SDPLin(hid_dim, hid_dim, gamma=gamma),
        #                             activation,
        #                             SDPLin(hid_dim, input_dim, gamma=gamma))
        self.net = nn.Sequential(   SDPLin(input_dim+1, hid_dim, gamma=gamma),
                                    activation,
                                    SDPLin(hid_dim, input_dim, gamma=gamma))

    def forward(self, x):
        return self.net(x)

class SimpleMap(nn.Module):
    def __init__(self, input_dim, hid_dim=128):
        super(SimpleMap, self).__init__()
        gamma = 10.0#(2.0)**(1/4)
        # input includes stacked time dimension
        self.activation = nn.ReLU()
        self.f1 = nn.Linear(input_dim, hid_dim)
        self.f2 = nn.Linear(hid_dim, input_dim)
        self.f3 = nn.Linear(input_dim, hid_dim)
        self.f4 = nn.Linear(hid_dim, input_dim)
        self.f5 = nn.Linear(input_dim, hid_dim)
        self.f6 = nn.Linear(hid_dim, input_dim)
        self.f7 = nn.Linear(input_dim, hid_dim)
        self.f8 = nn.Linear(hid_dim, input_dim)


    def forward(self, x):
        x = x + self.f2(F.relu(self.f1(x)))
        x = x + self.f4(F.relu(self.f3(x)))
        x = x + self.f6(F.relu(self.f5(x)))
        x = x + self.f8(F.relu(self.f7(x)))
        return x

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


class AddInterpolant(nn.Module):
    def __init__(self, state_dim, hid_dim=64):
        super(AddInterpolant, self).__init__()

        activation = nn.ReLU()
        self.state_dim = state_dim
        self.fnn = nn.Sequential(nn.Linear(state_dim*2+1, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, hid_dim),
                                    activation,
                                    nn.Linear(hid_dim, state_dim))

    def forward(self, x0, x1, t):
        t.requires_grad = True
        z = torch.cat((x0,x1,t),1)
        fnn = self.fnn(z)
        dt_fnn = vmap(jacrev(self.fnn))(z)[:,:,-1]
        xt = (1-t)*x0 + t*x1 + t*(1-t)*fnn
        dt_xt = x1 - x0 + (1-t)*fnn - t * fnn + t*(1-t)*dt_fnn

        return xt, dt_xt

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

def hole_barrier(x, c, r):
    x1 = x[:,0]
    x2 = x[:,1]

    return torch.square(x1-c[0]) + torch.square(x2-c[1]) - r**2

def hole_barrier_cond(x, c, r, V):
    x1 = x[:,0]
    x2 = x[:,1]

    grad_h_x1 = 2*(x1-c[0])
    grad_h_x2 = 2*(x2-c[1])

    return grad_h_x1 * V[:,0] + grad_h_x2 * V[:,1] + 10. * hole_barrier(x, c, r)

