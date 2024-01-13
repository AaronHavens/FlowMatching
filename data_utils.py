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

def generate_circle_flow(N):

	Y = np.zeros((N, 2))
	X = np.zeros((N, 3))

	# iterative sampling for now (for my sanity)
	for i in range(N):
		t = np.random.uniform(0,1)
		x_0 = np.random.uniform(-1.0,1.0, size=(2,))

		# uniform random sample on S1
		x_1 = np.random.normal(0,1, size=(2,))
		x_1 = x_1 / np.linalg.norm(x_1)

		xt = linear_traj(t, x_0, x_1)
		dx_dt = linear_traj_dt(t, x_0, x_1)

		X[i, :] = np.array([xt[0], xt[1], t])

		Y[i, :] = dx_dt


	return X, Y


def get_circle_dataset(N, batch_size=64):

	X, Y = generate_circle_flow(N)

	tensor_x = torch.Tensor(X) # transform to torch tensor
	tensor_y = torch.Tensor(Y)

	dataset = TensorDataset(tensor_x,tensor_y) # create datset and loader
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return dataloader


def circle_barrier(x):
	x1 = x[:,0]

	return -torch.square(x1) + 0.5**2

def circle_barrier_cond(x, V):
	x1 = x[:,0]
	grad_h_x1 = -2*x1


	return grad_h_x1 * V[:,0] + 10.*circle_barrier(x)

