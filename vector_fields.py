import torch
import torch.nn as nn


class simpleVF(nn.Module):
	def __init__(self, input_dim, hid_dim=128):
		super(simpleVF, self).__init__()
		# input includes stacked time dimension
		self.net = nn.Sequential(	nn.Linear(input_dim+1, hid_dim),
									nn.ReLU(),
									nn.Linear(hid_dim, hid_dim),
									nn.ReLU(),
									nn.Linear(hid_dim, hid_dim),
									nn.ReLU(),
									nn.Linear(hid_dim, input_dim))

	def forward(self, x):
		return self.net(x)



