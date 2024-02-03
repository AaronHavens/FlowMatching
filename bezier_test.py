import torch
from vector_fields import BezierInterpolant
import matplotlib.pyplot as plt
import numpy as np

N = 100
M = 100
state_dim = 2

interpolant = BezierInterpolant(state_dim)

x0 = torch.Tensor([[0.,0.]])
x1 = torch.randn((1,state_dim))
ts = np.linspace(0,1,N)
x_path = np.zeros((N, state_dim))

with torch.no_grad():
	for j in range(M):
		x1 = torch.randn((1,state_dim))
		for i,t in enumerate(ts):
			xt, _ = interpolant(x0, x1, t)
			print(i, xt)
			x_path[i,:] = xt.detach().numpy()[0,:]
		plt.plot(x_path[:,0],x_path[:,1])
		plt.scatter(x1.detach()[0,0],x1.detach()[0,1],c='r')

plt.scatter(x0.detach()[0,0],x0.detach()[0,1],c='r')
plt.show()