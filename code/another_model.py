#https://en.wikipedia.org/wiki/Intelligent_driver_model


import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import pdb

v0 = 30.
T = 1.5
a = 1.
b = 3.
delta = 4.
s0 = 2.
end_of_track = 600
t_steps = 100000


def x_dash(t, v):
	return v

def x(x0, v, t):
	return x0 + v*t

def s_star(v, delta_v):
	return s0 + v*T + (v*delta_v/(2*np.sqrt(a*b)))**2

def x_v_dash(x_v, t):
	# pdb.set_trace()
	x_v = x_v.reshape(2,-2)
	v = x_v[1,:]
	x_vec = x_v[0,:]
	delta_v = v.copy()

	s = x_vec.copy()

	for i in range(len(s)):
		if i == len(s)-1:
			delta_v[i] = v[0] - v[-1]
			s[i] =  -end_of_track%(x_vec[0] - x_vec[-1])
		else:
			delta_v[i] =  v[i] - v[i+1]
			s[i] =  x_vec[i] - x_vec[i+1]
	# print x_vec
	# print s
	# print delta_v
	# if (x_vec[-1] < 0.1): 
	# pdb.set_trace()
	x_v = np.concatenate(([v], [a*(1-(v/v0)**delta - (s_star(v, delta_v)/s)**2)]), axis=0).reshape(1,-1)[0]
	return x_v

v = np.ones(50) * 30
x_vec = np.linspace(0,490,50)
x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]

ts = np.linspace(0,1000,t_steps)

y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts)

fig, ax = plt.subplots(1,1, figsize=(8,8))
# pdb.set_trace()q
y_s = y_s.reshape(t_steps,-2,2)
# for car in range(50):
# 	ax.plot(ts, x_vec)
for car in range(50):
	# pdb.set_trace()
	print y_s[:,car,0]
	ax.plot(ts, y_s[:,car,0])
# print xs
plt.show()




