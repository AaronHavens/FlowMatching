import torch
import torch.nn as nn
from time import time
from torch.func import jacfwd, vmap

class InvertibleBlock(nn.Module):
    def __init__(self, input_dim, hid_dim=32, activation=nn.ReLU):
        super().__init__()

        self.input_dim = input_dim
        self.split1 = int(input_dim//2)
        self.split2 = int(input_dim//2 + input_dim%2)

        self.s1 = nn.Sequential(nn.Linear(self.split1, hid_dim),
                activation(),
                nn.Linear(hid_dim, self.split2))

        self.s2 = nn.Sequential(nn.Linear(self.split2, hid_dim),
                activation(),
                nn.Linear(hid_dim, self.split1))

        self.t1 = nn.Sequential(nn.Linear(self.split1, hid_dim),
                activation(),
                nn.Linear(hid_dim, self.split2))

        self.t2 = nn.Sequential(nn.Linear(self.split2, hid_dim),
                activation(),
                nn.Linear(hid_dim, self.split1))

    def forward(self, x, inverse=False):
        if not inverse:
            u1 = x[:self.split1]
            u2 = x[self.split1:-1]
            t = x[-1:]
            v1 = u1 * torch.exp(self.s2(u2)) + self.t2(u2)
            v2 = u2 * torch.exp(self.s1(v1)) + self.t1(v1)
            return torch.cat((v1, v2, t), axis=0)
        else:
            v1 = x[:self.split1]
            v2 = x[self.split1:-1]
            t = x[-1:]

            u2 = (v2 - self.t1(v1)) * torch.exp(-self.s1(v1))
            u1 = (v1 - self.t2(u2)) * torch.exp(-self.s2(u2))
            return torch.cat((u1, u2, t), axis=0)


class Diffeomorpism(nn.Module):
    def __init__(self, input_dim, hid_dim=32, n_layers=3):
        super().__init__()

        self.input_dim = input_dim
        activation = nn.Tanh
        self.n_layers = n_layers
        self.layers = nn.ModuleList([InvertibleBlock(input_dim, activation=activation) for j in range(n_layers)])
        

    def forward(self, x, inverse=False):
        if not inverse:
            for j in range(self.n_layers):
                x = self.layers[j](x)
            return x
        else:
            for j in reversed(range(self.n_layers)):
                x = self.layers[j](x, inverse=True)
            return x



class F_Related_VF(nn.Module):
    def __init__(self, input_dim, hid_dim=128, n_layers=3):
        super().__init__()
        # input includes stacked time dimension
        self.input_dim = input_dim
        self.F = Diffeomorpism(input_dim, n_layers=n_layers)
        #self.ones_x = torch.ones(input_dim+1, requires_grad=False)
        #self.ones_x[-1] = 0
        #self.ones_t = torch.zeros(input_dim+1, requires_grad=False)
        #self.ones_t[-1] = 1
        #self.DF = vmap(jacfwd(F, argnums=0))
        self.A = nn.Parameter(torch.rand(input_dim, input_dim+1))
        #self.A = torch.zeros(input_dim,input_dim)

    def get_A_tilde(self):
        I_t = torch.zeros(1,self.input_dim+1)
        return torch.cat((self.A, I_t),axis=0)

    def get_A_tilde_eval(self):
        A_tilde = self.get_A_tilde()
        A = torch.zeros(self.input_dim+2,self.input_dim+2)
        A[:self.input_dim+1,:self.input_dim+1] = A_tilde
        A[-2,-1] = 1
        return A

    def forward(self, y):
        x = vmap(self.F)(y, inverse=True)
        t = x[:,-1:]
        DF_x = vmap(jacfwd(self.F, argnums=0))(x)
        A = self.get_A_tilde()
        I_t = torch.zeros(3)
        I_t[-1] = 1
        Ax = nn.functional.linear(x, A) + I_t[None,:]
        #DF_x = torch.transpose(DF_x, 1, 2)
        #Y_y = torch.bmm(DF_x, Ax[:,:,None])
        #return Y_y[:,:2,0]
        return torch.einsum('bij,bj->bi', DF_x, Ax)[:,:2]
        #return torch.zeros_like(y)


    def time_t_flow(self, y, t=1):
        x = vmap(self.F)(y, inverse=True)
        ones = torch.ones(x.shape[0], 1)
        x = torch.cat((x,ones),axis=1)
        #print(x)
        A = self.get_A_tilde_eval()
        expA = torch.linalg.matrix_exp(A*t)
        yT = (vmap(self.F)(nn.functional.linear(x, expA)[:,:self.input_dim+1]))
        return yT[:,:2]
        #return (vmap(self.F)(x + self.ones_x*t**2/2))[:,:2]


# x_dim = 10
# bs = 2
# torch.manual_seed(0)
# F = Diffeomorpism(x_dim)
# VF = F_Related_VF(F, x_dim)

# Y = torch.ones(bs, x_dim)

# VF_Y = VF(Y)

# print(VF_Y)




