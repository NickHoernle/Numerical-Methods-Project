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

def a_IIDM(v,s,delta_v,params):
	a = params['a']
	delta = params['delta']
	v0 = params['v0']
	# Compute dv/dt (i.e. accelerations)
	return a*(1-(v/v0)**delta - (s_star(v, delta_v, params)/s)**2)

def a_CAH(s, v, vl, v_dash_l,params):
	a = params['a']
	delta = params['delta']
	v0 = params['v0']
	# Compute dv/dt (i.e. accelerations)
	a_tilde_l = np.minimum(v_dash_l, a)
	mask = ((vl*(v-vl))<=(-2*s*a_tilde_l))
	return ((np.square(v)*a_tilde_l/(np.square(vl) - 2*s*a_tilde_l) * mask)
					+ (a_tilde_l - (np.square(v-vl)*((v-vl)>0)/(2*s)))*(1-mask))

def a_ACC(s, v, vl, v_dash_l,params):
	c = params['c']
	b = params['b']

	delta_v =  v - vl
	a_iidm = a_IIDM(v, s, delta_v, params)
	a_cah = a_CAH(s, v, vl, v_dash_l, params)
	# print a_cah
	if (np.abs(s) < 0.1).any():
		pdb.set_trace()
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
	delta_v = v - vl
	# we require the change in speed of lead car,
	# as well as it's absolute speed
	vl_dot = (np.roll(params['y_s'][-2][:n_cars],1) - np.roll(params['y_s'][-1][:n_cars],1)/params['delta_t']
						if len(params['y_s']) > 1 else np.zeros(params['n_cars']))
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	s = x_vec - np.roll(x_vec,1)
	# put s for car zero within the bounds of the track
	s[0] -= (end_of_track - 1)
	# print s
	# Compute derivatives of position and velocity
	# dvdt = a_ACC(s, v, vl, vl_dot, params)
	# print params['y_s']
	dvdt = a_IIDM(v,s,delta_v,params)
	x_v = np.concatenate((v,dvdt))
	params['y_s'].append(x_v)
	return x_v

def runge_kutta_4(y_s, x_v_dash, t_k, h, params):
	x_v_vec_k = y_s[-1]
	k1=x_v_dash(x_v_vec_k,t_k, params)
	k2=x_v_dash(x_v_vec_k+.5*h*k1,t_k+.5*h, params)
	k3=x_v_dash(x_v_vec_k+.5*h*k2,t_k+.5*h, params)
	k4=x_v_dash(x_v_vec_k+h*k3,t_k+h, params)
	x_v_vec_k_next = x_v_vec_k + h/6. * (k1 + 2*k2 + 2*k3 + k4)
	return x_v_vec_k_next

if __name__ == '__main__':
	## Parameters ##
	params 									= dict()
	params['v0'] 						= 20.0 # desired velocity (in m/s) of vehicles in free traffic
	params['init_v'] 				= 5.0 # initial velocity
	params['T'] 						= 1.5 # Safe following time
	params['a'] 						= 1.0 # Maximum acceleration (in m/s^2)
	params['b'] 						= 3.0 # Comfortable deceleration (in m/s^2)
	params['delta'] 				= 4.0 # Acceleration exponent
	params['s0'] 						= 2.0 # minimum gap (in m)
	params['end_of_track'] 	= 600 # in m
	params['t_steps'] 			= 1000 # number of timesteps
	params['n_cars'] 				= 50 # number of vehicles
	params['total_time'] 		= 500 # total time (in s)
	params['c'] 						= 0.99 # correction factor
	params['delta_t'] 				= (0-params['total_time'])/float(params['t_steps'])

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

	# Assign initial velocities (30m/s)
	v = np.ones(n_cars) * init_v
	# Assign initial positions
	x_vec = np.linspace(0,end_of_track-end_of_track/5,n_cars)
	# reverse positions so that car 0 is leading
	x_vec = x_vec[::-1]
	# create 1D vector of positions followed by velocities
	x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]
	# time
	ts = np.linspace(0,total_time,t_steps)
	# Solve System of ODEs
	params['y_s'] = [x_v_vec]
	y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts, args=(params,))
	# y_s=[]
	# y_s.append(x_v_vec)
	# for i in range(1,len(ts)):
	# 	y_s.append(runge_kutta_4(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))


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

	ani = animation.FuncAnimation(fig, animate, range(t_steps), init_func=init, interval=1, blit=False)

	ax.set_ylabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()