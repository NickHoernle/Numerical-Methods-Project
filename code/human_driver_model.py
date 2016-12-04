# Human Driver Model

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import math
# import scipy.stats.norm
import matplotlib.animation as animation
import pdb
from intelligent_driver_model import a_IDM, s_star, x_dash, x

def interp(u,i):
	j = int(Tr/t_step)
	r = Tr/t_step - j
	u_interp = r*u[i-j-1] + (1-r)*u[i-j]
	# pdb.set_trace()
	return u_interp

def wiener_process(tau_tilde, params):
	dt = params['t_step']
	n_cars = params['n_cars']
	t_steps = params['t_steps']
	w0 = np.random.randn(n_cars)
	w = np.zeros((n_cars, t_steps))
	w[:, 0] = w0

	for time_step in range(1, int(t_steps)):
		w[:, time_step] = np.exp(-dt/tau_tilde)*w[:, time_step-1] + np.sqrt(2*dt/tau_tilde)*np.random.randn(n_cars)

	return w

def x_v_dash(x_v, t,params):
	# Compute derivatives of position and velocity
	t_step = params['t_step']
	t_steps = params['t_steps']
	sigma_r = params['sigma_r']
	sigma_a = params['sigma_a']
	Vs = params['Vs']
	w_s = params['w_s']
	w_l = params['w_l']
	w_a = params['w_a']
	x_v = x_v.reshape(2,-2)
	v = x_v[1,:]
	x_vec = x_v[0,:]
	## Follow the leader ##
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	s = np.roll(x_vec,1) - x_vec
	# put s for car zero within the bounds of the track
	s[0] += end_of_track
	# follow the HDM equation
	index = math.floor(t/t_step) if math.floor(t/t_step) < t_steps else t_steps-1
	s_est = s * np.exp(Vs * w_s[:, index])
	# Compute estimates v^est
	# Note that for vehicle i, v^est_l[i] is vehicle i's estimate of
	# vehicle i-1's speed
	v_l_est = np.roll(v,1) - s*sigma_r*w_l[:, index]
	# Note: Car i follows car i-1
	# Since this is a ring track, car 0 follows car n-1
	# for vehicle i, delta_v_est = v[i] - v_est[i-1]
	delta_v_est = v - v_l_est
	# Compute acceleration (with estimation error)
	dvdt = a_IDM(v,s_est,delta_v_est,params) + sigma_a*w_a[:, index]
	x_v = np.concatenate((v,dvdt))
	return x_v

def runge_kutta_4(x_v_vec_k, x_v_dash, t_k, h,params):
	k1=x_v_dash(x_v_vec_k,t_k,params)
	k2=x_v_dash(x_v_vec_k+.5*h*k1,t_k+.5*h,params)
	k3=x_v_dash(x_v_vec_k+.5*h*k2,t_k+.5*h,params)
	k4=x_v_dash(x_v_vec_k+h*k3,t_k+h,params)
	x_v_vec_k_next = x_v_vec_k + h/6. * (k1 + 2*k2 + 2*k3 + k4)
	return x_v_vec_k_next

if __name__ == '__main__':
	## Parameters ##
	params = dict()
	params['v0'] = 20.0 # desired velocity (in m/s) of vehicles in free traffic
	params['init_v'] = 5.0 # initial velocity
	params['T'] = 1.5 # Safe following time
	params['a'] = 2.0 # Maximum acceleration (in m/s^2)
	params['b'] = 3.0 # Comfortable deceleration (in m/s^2)
	params['delta'] = 4.0 # Acceleration exponent
	params['s0'] = 2.0 # minimum gap (in m)
	params['end_of_track'] = 600 # in m
	params['t_steps'] = 5000 # number of timesteps
	params['t_start'] = 0.0
	params['n_cars'] = 50 # number of vehicles
	params['total_time'] = 500 # total time (in s)
	params['t_step'] = (params['total_time'] - params['t_start'])/params['t_steps']
	params['Tr'] = 0.6 # Reaction time
	params['na'] = 5 	# number of look ahead cars
	params['Vs'] = 0.1 # Variation coefficient of gap estimation error
	params['sigma_r'] = 0.01 # estimation error for the inverse TTC
	params['sigma_a']= 0.1  # magnitude of acceleration noise
	params['tau_tilde'] = 20.0 # persistence time of estimation errors (in s)
	params['tau_a_tilde'] =  1.0 # persistence time of acceleration noise (in s)
	v0 = params['v0']
	init_v = params['init_v']
	T = params['T']
	a = params['a']
	b = params['b']
	delta = params['delta']
	s0 = params['s0']
	end_of_track = params['end_of_track']
	t_steps = params['t_steps']
	n_cars = params['n_cars']
	total_time = params['total_time']
	Tr = params['Tr']
	na = params['na']
	Vs = params['Vs']
	sigma_r = params['sigma_r']
	sigma_a = params['sigma_a']
	tau_tilde = params['tau_tilde']
	tau_a_tilde = params['tau_a_tilde']
	t_start = params['t_start']
	t_step = params['t_step']
	params['w_s'] = wiener_process(tau_tilde, params)
	params['w_l'] = wiener_process(tau_tilde, params)
	params['w_a'] = wiener_process(tau_a_tilde, params)
	n_leading = 1


	v = np.ones(n_cars) * v0 # Initial velocities (in m/s)
	x_vec = np.linspace(0,end_of_track-end_of_track/5,n_cars)	# Initial positions
	# reverse initial positions
	x_vec = x_vec[::-1]
	ts = np.linspace(t_start,total_time,t_steps) # time steps
	x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]

	index = math.floor(t/t_step) if math.floor(t/t_step) < t_steps else t_steps-1
	s_est = s * np.exp(Vs * w_s[:, index])
	v_est = v - s*omega_r*w_s[:, index]

	# pdb.set_trace()
	# x_v = intelligent_driver_model_1(v_est, v-v_est, s_est).reshape(1,-1)[0]
	# print "orig", x_v

	x_v = (intelligent_driver_model_1(v_est, v-v_est, s_est)).reshape(1,-1)[0] + \
	np.concatenate((np.zeros(n_cars), omega_a*w_l[:, index]), axis=0)

	if len(past_vs) > int(Tr/t_step)+1:
	# 	v_prog = interp(past_v_s,0) + Tr * interp(past_a_s,0)
		v_l_prog = interp(past_vs,0)
		s_prog = s_est - Tr * interp(past_delta_vs,0)
		# pdb.set_trace()
		c_idm = np.sum([1./j**2 for j in range(1, n_leading+1)])**-1
		int_term = 0
		s_prog_i = s_prog
		# pdb.set_trace()
		v_prog_i = np.concatenate((v_l_prog[1:], [v_l_prog[0]]), axis=0)
		int_term += human_driver_model_int(s_prog, v_l_prog, v_l_prog)

		for i in range(1,n_leading):
			s_prog_i += np.concatenate((s_prog[i:], s_prog[0:i]), axis=0)
			v_prog_i = np.concatenate((v_l_prog[i:], v_l_prog[0:i]), axis=0)

			int_term += human_driver_model_int(v_l_prog, v_l_prog-v_prog_i, s_prog)
		# pdb.set_trace()
		print "orig", x_v
		pdb.set_trace()
		x_v = human_driver_model_free(v_est, v-v_est, s_est).reshape(1,-1)[0] + \
		int_term.reshape(1,-1)[0]
		print "est", x_v

	# Runge Kutta isn't quite working, but I haven't looked into why
	y_s=[]
	v_s=[]
	y_s.append(x_v_vec)
	for i in range(1,len(ts)):
		y_s.append(runge_kutta_4(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1],params))
		v_s.append(y_s[-1][n_cars:])
	y_s = np.array(y_s)
	#y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts,args=(params,))

	# Plot position and velocity of each car
	fig, axes = plt.subplots(1,2, figsize=(16,8))
	# Plot positions over time
	for car in xrange(n_cars):
		axes[0].plot(ts, y_s[:,car])
	# Plot velocity over time
	for car in xrange(n_cars):
		axes[1].plot(ts, y_s[:,car+n_cars])
	axes[0].set_xlabel('Time')
	axes[0].set_ylabel('Position')
	axes[1].set_xlabel('Time')
	axes[1].set_ylabel('Velocity')
	plt.show()

	# Run a simulation of sorts
	# Plot Animation of cars on ring track
	r = (end_of_track/(2*np.pi))
	fx = lambda x_vec: r*np.sin((x_vec/end_of_track)*2*np.pi)
	fy = lambda x_vec: r*np.cos((x_vec/end_of_track)*2*np.pi)
	fig, ax = plt.subplots()
	t = np.arange(0, end_of_track, dtype=np.float32)
	t_pos = np.linspace(0, 2*np.pi, n_cars)
	line, = ax.plot(fx(t), fy(t))
	cars = ax.scatter(fx(x_vec), fy(x_vec), c='g')

	def animate(i):
		x_vec = y_s[i,:n_cars]
		x_vec = np.remainder(x_vec, end_of_track)
		new_pos = np.concatenate(([fx(x_vec)], [fy(x_vec)]), axis=0)
		cars.set_offsets(new_pos.T)
		return cars,

	def init():
		return cars,

	ani = animation.FuncAnimation(fig, animate, range(t_steps), init_func=init, interval=25, blit=False)

	ax.set_ylabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()
