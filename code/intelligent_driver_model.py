# Intelligent Driver Model as described in Traffic Flow Dynamics
# By Treiber et al.

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

def x_dash(t, v):
	return v

def x(x0, v, t):
	return x0 + v*t

def s_star(v, delta_v,params):
	s0 = params['s0']
	T = params['T']
	a = params['a']
	b = params['b']
	# Compute s^*
	zeros = np.zeros(len(delta_v))
	# take max of 0 and each elt of v*T + v*delta_v/(2*np.sqrt(a*b))
	return s0 + np.amax(np.vstack((zeros,v*T + v*delta_v/(2*np.sqrt(a*b)))),axis=0)

def a_IDM(v,s,delta_v,params):
	# compute acceleration for IDM model
	a = params['a']
	delta = params['delta']
	v0 = params['v0']
	# Compute dv/dt (i.e. accelerations)
	#print v0
	#print s
	return a*(1-(v/v0)**delta - (s_star(v, delta_v, params)/s)**2)

def compute_a_free(v,s,delta_v,params):
	# compute free acceleration
	a = params['a']
	delta = params['delta']
	v0 = params['v0']
	b = params['b']
	mask = (v<=v0)
	return mask*(a*(1-(v/v0)**delta))+(1-mask)*(-b*(1-(v0/v)**(a*delta/b)))

def a_IIDM(v,s,delta_v,params):
	# compute acceleration for IIDM model
	a = params['a']
	v0 = params['v0']
	# compute z
	z = s_star(v, delta_v, params)/s
	v_mask = (v<=v0)
	z_mask = (z>=1)
	a_free = compute_a_free(v,s,delta_v,params)
	dvdt = np.zeros(v.shape)
	# v <= v0 and z >= 1
	dvdt += v_mask * z_mask * (a*(1-z**2))
	# v <= v0 and z < 1
	dvdt += v_mask * (1-z_mask) * (a_free*(1-z**(2*a/a_free)))
	# v > v0 and z >= 1
	dvdt += (1-v_mask) * z_mask * (a_free+a*(1-z**2))
	# v > v0 and z < 1
	dvdt += (1-v_mask) * z_mask * (a_free)
	return dvdt

def a_CAH(s, v, vl, v_dash_l,params):
	a = params['a']
	delta = params['delta']
	v0 = params['v0']
	# Compute dv/dt (i.e. accelerations)
	a_tilde_l = np.minimum(v_dash_l, a)
	mask = ((vl*(v-vl))<=(-2*s*a_tilde_l))
	return ((np.square(v)*a_tilde_l/(np.square(vl) - 2*s*a_tilde_l) * mask)
					+ (a_tilde_l - (np.square(v-vl)*((v-vl)>0)/(2*s)))*(1-mask))

def a_ACC(s, v, vl, a_iidm, params):
	c = params['c']
	b = params['b']
	# get dvdt for leading cars
	v_dash_l = np.roll(a_iidm,1)
	a_cah = a_CAH(s, v, vl, v_dash_l, params)
	mask = a_iidm >= a_cah
	return ((a_iidm * mask)
					+ ((1-c)*a_iidm + c*(a_cah + b*np.tanh((a_iidm-a_cah)/b)))*(1-mask))

def x_v_dash(x_v, t, params):
	# Compute derivative of position and velocity
	x_v = x_v.reshape(2,-2)
	# get velocities
	v = x_v[1,:]
	# get positions
	x_vec = x_v[0,:]
	## Follow the leader ##
	# Note: Car i follows car i-1
	# Since this is a ring track, car 0 follows car n-1
	# for vehicle i, delta_v = v[i] - v[i-1]
	vl = np.roll(v,1)
	vl[0] = float('inf')
	# Compute difference in speeds between current car and leading car
	delta_v = v - vl

	# Compute gap
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	# put s for car zero within the bounds of the track
	s = np.roll(x_vec,1) - x_vec
	s[0] += params['end_of_track']

	s[0] = float('inf')
	# Compute acceleration
	if params['IDM_model_num'] == 0:
		# Standard IDM
		dvdt = a_IDM(v,s,delta_v,params)
	else:
		# compute Improved IDM acceration function
		# Note: we compute this acceleration function for both IIDM and ACC
		# For ACC, this serves as dvdt
		dvdt = a_IIDM(v,s,delta_v,params)
	if params['IDM_model_num'] == 2:
		# AAC IDM
		dvdt = a_ACC(s, v, vl, dvdt, params)
	if t <= 200:
		v[0] = 15
	if t > 200:
		dvdt[0] = 1
	if t > 240:
		dvdt[0] = -1
	if t > 280:
		dvdt[0] = 0
	x_v = np.concatenate((v,dvdt))
	# params['y_s'].append(x_v)
	return x_v

def runge_kutta_5(x_v_vec_k, x_v_dash, t_k, h, params):
	k1 = x_v_dash(x_v_vec_k,t_k, params)
	k2 = x_v_dash(x_v_vec_k + 1./4*h*k1,t_k+1./4*h, params)
	k3 = x_v_dash(x_v_vec_k + 3./32*h*k1 + 9./32*h*k2,t_k+3./8*h, params)
	k4 = x_v_dash(x_v_vec_k + 1932./2197*h*k1 -7200./2197*h*k2 + 7296./2197*h*k3,t_k+12./13*h, params)
	k5 = x_v_dash(x_v_vec_k + 439./216*h*k1 -8.*h*k2 + 3680./513*h*k3 - 845./4104*h*k4,t_k*h, params)
	k6 = x_v_dash(x_v_vec_k - 8./27*h*k1 + 2.*h*k2 -3544./2565*h*k3 + 1859./4104*h*k4 - 11./40*h*k5,t_k*1./2*h, params)
	x_v_vec_k_next = x_v_vec_k + h * (15./135*k1 + 6656./12825*k3 + 28561./56430*k4 - 9./50*k5 + 2./55*k6)
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def runge_kutta_4(x_v_vec_k, x_v_dash, t_k, h, params):
	k1=x_v_dash(x_v_vec_k,t_k, params)
	k2=x_v_dash(x_v_vec_k+.5*h*k1,t_k+.5*h, params)
	k3=x_v_dash(x_v_vec_k+.5*h*k2,t_k+.5*h, params)
	k4=x_v_dash(x_v_vec_k+h*k3,t_k+h, params)
	x_v_vec_k_next = x_v_vec_k + h/6. * (k1 + 2*k2 + 2*k3 + k4)
	# store the derivative at each timestep
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def runge_kutta_3(x_v_vec_k, x_v_dash, t_k, h, params):
	k1=x_v_dash(x_v_vec_k,t_k,params)
	k2=x_v_dash(x_v_vec_k+.5*h*k1, t_k+.5*h, params)
	k3=x_v_dash(x_v_vec_k-h*k1+2*h*k2,t_k+h, params)
	x_v_vec_k_next = x_v_vec_k + h/6. * (k1 + 4*k2 + k3)
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def heuns_method(x_v_vec_k, x_v_dash, t_k, h, params):
	k1=x_v_dash(x_v_vec_k,t_k,params)
	k2=x_v_dash(x_v_vec_k+h*k1, t_k*h, params)
	x_v_vec_k_next = x_v_vec_k + h/2. * (k1 + k2)
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def fwd_euler(x_v_vec_k, x_v_dash, t_k, h, params):
	k1=x_v_dash(x_v_vec_k,t_k,params)
	x_v_vec_k_next = x_v_vec_k + h * (k1)
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def run_simulation(params):
	v = np.ones(params['n_cars']) * params['init_v']
	# Assign initial positions
	x_vec = np.linspace(0,params['end_of_track']-params['end_of_track']/6,params['n_cars'])
	# reverse positions so that car 0 is leading
	x_vec = x_vec[::-1]
	# create 1D vector of positions followed by velocities
	x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]
	# time
	ts = np.linspace(0,params['total_time'],params['t_steps'])
	# Solve System of ODEs
	#params['y_s'] = [x_v_vec]
	#y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts, args=(params,))
	y_s=[]
	y_s.append(x_v_vec)
	for i in range(1,len(ts)):
		y_s.append(runge_kutta_4(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
	y_s = np.array(y_s)

	return y_s


if __name__ == '__main__':
	## Parameters ##
	params 									= dict()
	params['v0'] 						= 30.0 # desired velocity (in m/s) of vehicles in free traffic
	params['init_v'] 				= 5.0 # initial velocity
	params['T'] 						= 1.0 # Safe following time
	params['a'] 						= 2.0 # Maximum acceleration (in m/s^2)
	params['b'] 						= 3.0 # Comfortable deceleration (in m/s^2)
	params['delta'] 				= 4.0 # Acceleration exponent
	params['s0'] 						= 2.0 # minimum gap (in m)
	params['end_of_track'] 	= 600 # in m
	params['t_steps'] 			= 1000 # number of timesteps
	params['n_cars'] 				= 50 # number of vehicles
	params['total_time'] 		= 500 # total time (in s)
	params['c'] 						= 0.99 # correction factor
	params['t_start'] = 0.0
	params['t_step'] = (params['total_time']-params['t_start'])/float(params['t_steps'])
	params['x_v_dash'] 			= [2*np.zeros(params['n_cars'])]

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
	# Indicates what model to use
	# 0 == IDM
	# 1 == IIDM
	# 2 == ACC
	for model in xrange(3):
		params['IDM_model_num'] = model

		# Assign initial velocities (30m/s)
		v = np.ones(n_cars) * init_v
		# Assign initial positions
		x_vec = np.linspace(0,end_of_track-end_of_track/6,n_cars)
		# reverse positions so that car 0 is leading
		x_vec = x_vec[::-1]
		# create 1D vector of positions followed by velocities
		x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]
		# time
		ts = np.linspace(0,total_time,t_steps)
		# Solve System of ODEs
		#params['y_s'] = [x_v_vec]
		#y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts, args=(params,))
		y_s=[]
		y_s.append(x_v_vec)
		for i in range(1,len(ts)):
			y_s.append(runge_kutta_3(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))

		# Plot position and velocity of each car
		fig, axes = plt.subplots(1,2, figsize=(16,8))
		# Plot positions over time
		for car in xrange(n_cars):
			axes[0].plot(ts, y_s[:,car])
		# Plot velocity over time
		for car in xrange(n_cars):
			axes[1].plot(ts, y_s[:,car+n_cars])
		axes[0].set_xlabel('Time')
		axes[0].set_ylabel('Displacement')
		axes[1].set_xlabel('Time')
		axes[1].set_ylabel('Velocity')
		plot_out_name = "../figures/displacement_and_velocity_plot_model{}.pdf".format(params['IDM_model_num'])
		plt.savefig(plot_out_name,
				orientation='landscape',format='pdf',edgecolor='black')
		# plt.close()
		plt.show()

		# Run a simulation of sorts
		# Plot Animation of cars on ring track
		# r = (end_of_track/(2*np.pi))
		# fx = lambda x_vec: r*np.sin((x_vec/end_of_track)*2*np.pi)
		# fy = lambda x_vec: r*np.cos((x_vec/end_of_track)*2*np.pi)
		# fig, ax = plt.subplots()
		# t = np.arange(0, end_of_track, dtype=np.float32)
		# t_pos = np.linspace(0, 2*np.pi, n_cars)
		# line, = ax.plot(fx(t), fy(t))
		# cars = ax.scatter(fx(x_vec), fy(x_vec), c='g')

		# def animate(i):
		# 	x_vec = y_s[i,:n_cars]
		# 	x_vec = np.remainder(x_vec, end_of_track)
		# 	new_pos = np.concatenate(([fx(x_vec)], [fy(x_vec)]), axis=0)
		# 	cars.set_offsets(new_pos.T)
		# 	return cars,

		# def init():
		# 	return cars,

		# ani = animation.FuncAnimation(fig, animate, range(t_steps), init_func=init, interval=1, blit=False)

		# ax.set_ylabel('$x$')
		# ax.set_ylabel('$y$')
		# plt.show()
		# plt.close()
