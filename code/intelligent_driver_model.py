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

def s_star(v, delta_v):
	# Compute s^*
	zeros = np.zeros(len(delta_v))
	# take max of 0 and each elt of v*T + v*delta_v/(2*np.sqrt(a*b))
	return s0 + np.amax(np.vstack((zeros,v*T + v*delta_v/(2*np.sqrt(a*b)))),axis=0)

def intelligent_driver_model_1(v, a, v0, delta, delta_v, s):
	# Compute derivatives of position and velocity
	dxdt = v
	dvdt = a*(1-(v/v0)**delta - (s_star(v, delta_v)/s)**2)
	return np.concatenate(([dxdt], [dvdt]), axis=0)

def x_v_dash(x_v, t):
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
	delta_v = v - np.roll(v,1)
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	s = np.roll(x_vec,1) - x_vec
	# put s for car zero within the bounds of the track
	s[0] += end_of_track
	# Compute derivatives of position and velocity
	x_v = intelligent_driver_model_1(v, a, v0, delta, delta_v, s).reshape(1,-1)[0]
	return x_v

if __name__ == '__main__':
	## Parameters ##
	v0 = 20 # desired velocity (in m/s) of vehicles in free traffic (also initial v)
	T = 1.5 # Safe following time
	a = 1.0 # Maximum acceleration (in m/s^2)
	b = 3.0 # Comfortable deceleration (in m/s^2)
	delta = 4.0 # Acceleration exponent
	s0 = 2.0 # minimum gap (in m)
	end_of_track = 600 # in m
	t_steps = 1000 # number of timesteps
	n_cars = 50 # number of vehicles
	total_time = 500 # total time (in s)

	# Assign initial velocities (30m/s)
	v = np.ones(n_cars) * 5
	# Assign initial positions
	x_vec = np.linspace(0,end_of_track-end_of_track/5,n_cars)
	# reverse positions so that car 0 is leading
	x_vec = x_vec[::-1]
	# create 1D vector of positions followed by velocities
	x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]
	# time
	ts = np.linspace(0,total_time,t_steps)
	# Solve System of ODEs
	y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts)

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