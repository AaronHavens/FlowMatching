import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


w = 1.0
T = 1000.0
steps = 10000

def dxdt(x,t):
	if abs(x[0]) > 1:
		ddx = -1*x[0] - 1*x[1]
	else:
		ddx = 0

	if x[0] == 0 and x[1] == 0:
		ddx += w
	elif x[0] >= 0 and x[1] > 0:
		ddx += w
	elif x[0] >= 0 and x[1] < 0:
		ddx -= w
	elif x[0] <= 0 and x[1] < 0:
		ddx -= w
	elif x[0] <= 0  and x[1] > 0:
		ddx += w

	return np.array([x[1], ddx])



x0 = np.array([0., 0.])

sol = odeint(dxdt, x0, np.linspace(0,T,steps))

linf = np.zeros(steps)
running_max = 0.0
for i in range(steps):
	max_i = max(abs(sol[i,0]), abs(sol[i,1]))
	if max_i > running_max:
		running_max = max_i
	linf[i] = running_max
plt.subplot(121)
plt.plot(sol[:,0],label='pos')
plt.plot(sol[:,1], label='vel')
plt.xlabel('x(t)')
plt.ylabel('dx(t)/dt')
plt.legend()
plt.subplot(122)
plt.plot(linf,label='truncated l_inf norm')
plt.xlabel('time')
plt.ylabel('truncated l-inf norm')
plt.legend()
plt.show()

