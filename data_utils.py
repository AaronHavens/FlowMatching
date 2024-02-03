import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

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


def generate_circle_flow(interpolant, batch_size=1):
	t = torch.rand(size=(batch_size,1))
	x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
	x1 = torch.randn(size=(batch_size,2))
	x1 = x1/torch.norm(x1,dim=-1, keepdim=True)

	X, Y = interpolant(x0,x1,t)
	X = torch.cat((X,t), 1)
	return X, Y

def generate_grid_flow(batch_size=1):

	Y = np.zeros((batch_size, 2))
	X = np.zeros((batch_size, 3))
	X_aux = np.zeros((batch_size,2))
	# iterative sampling for now (for my sanity)
	for i in range(batch_size):
		t = np.random.uniform(0,1)
		
		#x_0 = np.random.normal(0,1,size=(2,))*0.2
		#r_0 = np.random.normal(0,1)**(1/2.)
		#x_0 = x_0 / np.linalg.norm(x_0)

		x_0 = np.random.uniform(0,1,size=(2,))

		grid_range = [(0.,0.),(0,2./5),(0, 4./5),
						(1./5, 1./5), (1./5, 3./5),
						(2./5,0.),(2./5,2./5),(2./5, 4./5),
						(3./5, 1./5), (3./5, 3./5),
						(4./5,0.),(4./5, 2./5),(4./5, 4./5)]
		j = np.random.randint(0,13)
		x_11 = np.random.uniform(grid_range[j][1], grid_range[j][1]+1./5)
		x_12 = np.random.uniform(grid_range[j][0], grid_range[j][0]+1./5)
		x_1 = np.array([x_11,x_12])
		#X_aux[i,:] = x_1 
		# # uniform random sample on unit Disc
		# x_1 = np.random.normal(0,1, size=(2,))
		# r_1 = np.random.uniform(0,1)**(1./2.)
		# x_1 = x_1 / np.linalg.norm(x_1)*r_1

		xt = linear_traj(t, x_0, x_1)
		dx_dt = linear_traj_dt(t, x_0, x_1)

		X[i, :] = np.array([xt[0], xt[1], t])

		Y[i, :] = dx_dt

	#plt.scatter(X_aux[:,0], X_aux[:,1],s=0.5)
	#plt.show()
	return torch.Tensor(X), torch.Tensor(Y)






