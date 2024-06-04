import torch
from f_related_utils import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)


vf = F_Related_VF(2, n_layers=1, hid_dim=8)

def simulate(x0, steps=50):
    def dxdt(y, t):
        y_ = torch.Tensor([[y[0], y[1], t]])
        return vf(y_).detach().numpy()[0]

    sol = odeint(dxdt, x0, np.linspace(0,1,steps))

    return sol



steps = 50
T = 1.0
X0 = np.random.normal(0,T,size=(2,))

sol_ode = simulate(X0, steps=steps)
t = np.linspace(0,T,steps)

#print(sol_ode)

X0 = torch.tensor(X0)[None, :]
X0 = torch.cat((X0,torch.zeros(1,1)),axis=1).float()

sol_exp = np.zeros((steps,2))
for j in range(steps):
	sol_exp[j,:] = vf.time_t_flow(X0, t[j]).detach().numpy()




plt.plot(sol_ode[:,0], sol_ode[:,1], label='ode solution')
plt.plot(sol_exp[:,0], sol_exp[:,1], label='exp solution')
plt.legend()

plt.show()