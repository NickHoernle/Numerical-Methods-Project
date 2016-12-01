#https://en.wikipedia.org/wiki/Intelligent_driver_model


import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
# import scipy.stats.norm
import pdb

v0 = 30. 				# initial velocity of cars
T = 1.5 				# Safe following time
Tr = 0.6 				# Reaction time
na = 5 					# number of look ahead cars
Vs = .1 				# Variation coefficient of gap estimation error
omega_r = 0.01	# estimation error for the inverse TTC
omega_a = 0.1 	# magnitude of acceleration noise
tau_tilde = 20	# persistence time of estimation errors
tau_a_tilde =  1 # persistence time of acceleration noise
a = 1.
b = 3.
delta = 4.
s0 = 2.
end_of_track = 600
t_steps = 10000
n_cars = 50.
t_start = 0.
t_end = 1000.
t_step = (t_end - t_start)/t_steps
ts = np.linspace(t_start,t_end,t_steps) # time steps

def x_dash(t, v):
	return v

def x(x0, v, t):
	return x0 + v*t

def s_star(v, delta_v):
	return s0 + v*T + (v*delta_v/(2*np.sqrt(a*b)))**2

def intelligent_driver_model_1(v, delta_v, s):
	return np.concatenate(([v], [a*(1-(v/v0)**delta - (s_star(v, delta_v)/s)**2)]), axis=0)

def weiner_process(dt, tau_tilde):
	w0 = np.random.randn(n_cars)
	w = np.zeros((n_cars, t_steps*1e2))
	w[:, 0] = w0

	for time_step in range(1, int(t_steps*1e2)):
		w[:, time_step] = np.exp(-dt/tau_tilde)*w[:, time_step-1] + np.sqrt(2*dt/tau_tilde)*np.random.randn(n_cars)

	return w

w_s = weiner_process(t_step, tau_tilde)
w_l = weiner_process(t_step, tau_a_tilde)

def x_v_dash(x_v, t):
	# pdb.set_trace()
	x_v = x_v.reshape(2,-2)
	v = x_v[1,:]
	x_vec = x_v[0,:]
	delta_v = v.copy()

	s = x_vec.copy()

	for i in range(len(s)):
		# follow the leader
		if i == len(s)-1:
			delta_v[i] = v[0] - v[-1]
			s[i] = x_vec[-1] - x_vec[0] - end_of_track
		else:
			delta_v[i] =  v[i] - v[i+1]
			s[i] =  x_vec[i] - x_vec[i+1]

	index = np.where(ts-t<1e-2)[0][0]
	s_est = s * np.exp(Vs * w_s[:, index])

	v_est = v - s*omega_r*w_l[:, index]

	# the magic happens here. We have
	x_v = intelligent_driver_model_1(v_est, v-v_est, s_est).reshape(1,-1)[0]
	return x_v

v = np.ones(50) * 30
x_vec = np.linspace(0,490,50)
x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]

# ts = np.linspace(0,1000,t_steps)

y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts)

# fig, ax = plt.subplots(1,1, figsize=(8,8))

# for car in range(50):
# 	ax.plot(ts, y_s[:,car])
# plt.show()




