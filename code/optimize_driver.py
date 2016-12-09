import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
from intelligent_driver_model import *
from human_driver_model import *

def runge_kutta_4_driverparams(x_v_vec_k, x_v_dash, t_k, h, params, past, driveridx, driverparams):
	k1=x_v_dash(x_v_vec_k,t_k, params,driveridx, driverparams)
	k2=x_v_dash(x_v_vec_k+.5*h*k1,t_k+.5*h, params,driveridx, driverparams)
	k3=x_v_dash(x_v_vec_k+.5*h*k2,t_k+.5*h, params,driveridx, driverparams)
	k4=x_v_dash(x_v_vec_k+h*k3,t_k+h, params,driveridx, driverparams)
	x_v_vec_k_next = x_v_vec_k + h/6. * (k1 + 2*k2 + 2*k3 + k4)
	# store the derivative at each timestep
	params['x_v_dash'].append(k1)
	return x_v_vec_k_next

def x_v_dash_driver(x_v, t, params, driveridx, driverparams):
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
	# vl[0] = float('inf')
	# Compute difference in speeds between current car and leading car
	delta_v = v - vl

	# Compute gap
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	# put s for car zero within the bounds of the track
	s = np.roll(x_vec,1) - x_vec
	s[0] += params['end_of_track']

	# s[0] = float('inf')
	# Compute acceleration
	if params['IDM_model_num'] == 0:
		# Standard IDM
		dvdt = a_IDM(v,s,delta_v,params)
        dvdt_2 = a_IDM(v,s,delta_v,driverparams)
        dvdt[driveridx] = dvdt_2[driveridx]
	else:
		# compute Improved IDM acceration function
		# Note: we compute this acceleration function for both IIDM and ACC
		# For ACC, this serves as dvdt
		dvdt = a_IIDM(v,s,delta_v,params)
        dvdt_2 = a_IIDM(v,s,delta_v,driverparams)
        dvdt[driveridx] = dvdt_2[driveridx]
	if params['IDM_model_num'] == 2:
		# AAC IDM
		dvdt = a_ACC(s, v, vl, dvdt, params)
        dvdt_2 = a_ACC(s, v, vl, dvdt,driverparams)
        dvdt[driveridx] = dvdt_2[driveridx]
	# if t <= 200:
	# 	v[0] = 15
	# if t > 200:
	# 	dvdt[0] = 1
	# if t > 240:
	# 	dvdt[0] = -1
	# if t > 280:
	# 	dvdt[0] = 0
	x_v = np.concatenate((v,dvdt))
	# params['y_s'].append(x_v)
	return x_v

def x_v_dash_driver_human(x_v, t,params,driveridx, driverparams):
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
    dvdt_2 = a_IDM(v,s_est,delta_v_est,driverparams) + sigma_a*w_a[:, index]
    dvdt[driveridx] = dvdt_2[driveridx]
	x_v = np.concatenate((v,dvdt))
	return x_v

def x_v_dash2_driver_human(x_v, t,params,past,driveridx, driverparams):
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
	n_a = params['n_a']
	n_cars = params['n_cars']
	Tr = params['Tr']
	## Follow the leader ##
	# Compute true gap
	# for vehicle i, s = x_vec[i-1] - x_vec[i]
	s = np.roll(x_vec,1) - x_vec
	# put s for car zero within the bounds of the track
	s[0] += end_of_track
	# Compute index of this timestep
	index = int(math.floor(t/t_step)) if math.floor(t/t_step) < t_steps else t_steps-1
	# compute estimate of gap
	s_est = s * np.exp(Vs * w_s[:, index])
	# Compute estimates v^est
	# Note that for vehicle i, v^est_l[i] is vehicle i's estimate of
	# vehicle i-1's speed
	v_l_est = np.roll(v,1) - s*sigma_r*w_l[:, index]

	# update history
	past['past_v_l_est_s'].append(v_l_est)
	# Note: Car i follows car i-1
	# Since this is a ring track, car 0 follows car n-1
	# for vehicle i, delta_v_est = v[i] - v_est[i-1]
	delta_v_est = v - v_l_est
	past['past_delta_v_est_s'].append(delta_v_est)
	# update history
	past_v_l_est_s = np.array(past['past_v_l_est_s'])
	past_v_est_s = np.roll(past_v_l_est_s,-1)
	past_dvdt = np.array(past['past_dvdt'])
	past_delta_v_est_s = np.array(past['past_delta_v_est_s'])


	# Compute acceleration
	if past_v_est_s.shape[0] > int(Tr/t_step)+1:
		# compute v^prog_l
		v_l_prog = interp(past_v_l_est_s,0,params)
		v_prog = interp(past_v_est_s,0,params) + Tr * interp(past_dvdt,0,params)
		# compute s^prog
		#print s_est
		s_prog = s_est - Tr * interp(past_delta_v_est_s,0,params)
		# compute c_idm
		c_idm = np.sum([1./j**2 for j in xrange(1, n_a+1)])**(-1)
		free_term = a_IDM_free(v, params)
		dvdt = np.zeros(v.shape)
		# calculate the acceleration of each vehicle one at a time
		# As written in Eq 12.20 of the textbook
		for alpha in xrange(n_cars):
            if alpha == driveridx:
                free_term_alpha = free_term[alpha]
    			int_term = 0.0
    			v_alpha_prog = np.array([v_prog[alpha]])
    			for beta in xrange(alpha-n_a,alpha):
    				# we want to sum from beta+1 to alpha
    				# precompute these indices to avoid slicing issues
    				sum_idxs = np.arange(beta+1,alpha+1)
    				#print sum_idxs
    				s_alpha_beta_prog = np.array([np.sum(s_prog[sum_idxs])])
    				#print s_alpha_beta_prog
    				v_beta_prog = np.array([v_l_prog[beta]])
    				int_term += a_IDM_int(v_alpha_prog,s_alpha_beta_prog,v_alpha_prog-v_beta_prog,driverparams)
    			dvdt[alpha] = free_term_alpha + c_idm*int_term
            else:
    			free_term_alpha = free_term[alpha]
    			int_term = 0.0
    			v_alpha_prog = np.array([v_prog[alpha]])
    			for beta in xrange(alpha-n_a,alpha):
    				# we want to sum from beta+1 to alpha
    				# precompute these indices to avoid slicing issues
    				sum_idxs = np.arange(beta+1,alpha+1)
    				#print sum_idxs
    				s_alpha_beta_prog = np.array([np.sum(s_prog[sum_idxs])])
    				#print s_alpha_beta_prog
    				v_beta_prog = np.array([v_l_prog[beta]])
    				int_term += a_IDM_int(v_alpha_prog,s_alpha_beta_prog,v_alpha_prog-v_beta_prog,params)
    			dvdt[alpha] = free_term_alpha + c_idm*int_term
	else:
		# Compute acceleration (with estimation error)
		dvdt = a_IDM(v,s_est,delta_v_est,params) + sigma_a*w_a[:, index]
        dvdt_2 = a_IDM(v,s_est,delta_v_est,driverparams) + sigma_a*w_a[:, index]
        dvdt[driveridx] = dvdt_2[driveridx]
	x_v = np.concatenate((v,dvdt))
	past['past_dvdt'].append(dvdt)
	return x_v

# define a wrapper function to determine what cost is being calculated
# def cost_wrapper(optimize_params, cost_func, optimize_variables, params):
# 	for var, param in zip(optimize_variables, optimize_params):
# 		params[var] = param
#
# 	simulation_result = intelligent_driver_model.run_simulation(params)
# 	# pdb.set_trace()
# 	x_v_dash = np.array(params['x_v_dash'])
#
# 	displacement = simulation_result[:,:params['n_cars']].T
# 	velocity = simulation_result[:,params['n_cars']:].T
# 	acceleration = x_v_dash[:,params['n_cars']:].T
#
# 	# We should be able to write cost functions based on these three variables
# 	c = cost_func(displacement, velocity, acceleration, params)
# 	return c
#
# # Define the cost functions:
#
# # Cost Function 1
# # Minimizing the total travel time -> Page 419
# # Note: Minimising the travel time is the same as maximising the
# # displacement across the time interval
# def maximise_displacement(displacement, velocity, acceleration, params):
# 	return 1-np.sum(displacement[:,-1])/float(params['n_cars'])
#
# # Cost Function 2
# # Maximizing the driving comfort (minimise discomfort) -> Page 419
# def comfort_cost(displacement, velocity, acceleration, params):
# 	tau0 = 	1. #s -> characteristic time
# 	a0 = 		1. # m/s**2
# 	T0 = 		1. # arbitrary normalisation constant
#
# 	# enforce smooth boundary condition
# 	acceleration_dash = np.concatenate((np.zeros((params['n_cars'],1)), np.diff(acceleration)/params['t_step']), axis=1)
# 	integrand = np.sum(acceleration**2 + tau0**2*acceleration_dash**2, axis=0)
# 	cost = 1/(T0*a0**2)*sp.integrate.trapz(
# 									y=integrand,
# 									axis=0
# 								)
# 	return cost
#
# def comfort_cost_nick(displacement, velocity, acceleration, params):
# 	cost = np.sum(np.abs(acceleration[:,-1]))/float(params['n_cars'])
# 	return cost
#
# # Cost Function 3
# # Maximise average velocity
# def maximise_velocity(displacement, velocity, acceleration, params):
# 	return 1-np.sum(velocity[:,-1])/float(params['n_cars'])
#
# # Cost Function 4
# # Combination model
# def combination_model(displacement, velocity, acceleration, params):
# 	return maximise_velocity(displacement, velocity, acceleration, params) + maximise_displacement(displacement, velocity, acceleration, params)/100. +comfort_cost_nick(displacement, velocity, acceleration, params)



if __name__ == '__main__':
	## Parameters ##
	params = dict()
	params['v0'] = 50.0 # desired velocity (in m/s) of vehicles in free traffic
	params['init_v'] = 5.0 # initial velocity
	params['T'] = 2.5 # Safe following time
	# Maximum acceleration (in m/s^2)
	# Note: a is sensitive for the HDM model with anticipation
	# This is because c_idm is used to scale the acceleration interaction
	# and a multiplies each term in the interaction sum
	# SEE PAGE 216 in the book
	# Ex: a=2.0 is unstable
	# Ex: a=1.0 is stable
	params['a'] = 1.0
	params['T_idm'] = 1.
	params['a_idm'] = 1.0
	params['v0_idm'] = 50.0
	params['b'] = 3.0 # Comfortable deceleration (in m/s^2)
	params['delta'] = 4.0 # Acceleration exponent
	params['s0'] = 10.0 # minimum gap (in m)
	params['end_of_track'] = 1000 # in m
	params['t_steps'] = 1000 # number of timesteps
	params['t_start'] = 0.0
	params['n_cars'] = 50 # number of vehicles
	params['total_time'] = 300 # total time (in s)
	params['c'] = 0.99 # correction factor
	params['t_step'] = (params['total_time'] - params['t_start'])/params['t_steps']
	params['Tr'] = .6 # Reaction time
	params['Vs'] = 0.1 # Variation coefficient of gap estimation error
	params['sigma_r'] = 0.01 # estimation error for the inverse TTC
	params['sigma_a']= 0.01  # magnitude of acceleration noise
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
	Vs = params['Vs']
	sigma_r = params['sigma_r']
	sigma_a = params['sigma_a']
	tau_tilde = params['tau_tilde']
	tau_a_tilde = params['tau_a_tilde']
	t_start = params['t_start']
	t_step = params['t_step']
	params['n_a'] = 3 # number of cars ahead that the driver is aware of
	params['w_s'] = wiener_process(tau_tilde, params)
	params['w_l'] = wiener_process(tau_tilde, params)
	params['w_a'] = wiener_process(tau_a_tilde, params)
	params['j'] = int(Tr/t_step) # number of time steps in reaction time
	params['r'] = Tr/t_step - params['j'] # fractional part of Tr/t_step
	params['IDM_model_num'] = 2
	past = dict()
	past['past_v_l_est_s'] = deque(maxlen=params['j']+2)
	past['past_delta_v_est_s'] = deque(maxlen=params['j']+2)
	past['past_dvdt'] = deque(maxlen=params['j']+2)

    solution = sp.optimize.minimize(cost_wrapper, [5.] , args=(combination_model, ['v0'], params), bounds=[(2., 500.)])
	# solution = sp.optimize.minimize(cost_wrapper, [5.] , args=(maximise_displacement, ['v0'], params), bounds=[(2., 100.)])

	print solution.x

	# IDM_pct = .3
	# np.random.seed(0)
	# indices = np.random.permutation(range(2,n_cars))
	# IDM_idx = sorted(np.concatenate(([1,2],indices[:int(IDM_pct*n_cars)],[n_cars+1, n_cars+2], [i + n_cars for i in indices[:int(IDM_pct*n_cars)]]), axis=0))
	# HDM_idx = sorted(np.concatenate(([0],indices[int(IDM_pct*n_cars):],[n_cars],[i + n_cars for i in indices[int(IDM_pct*n_cars):]]), axis=0))
	# params['n_IDM_cars'] = int(IDM_pct*n_cars)
	# params['n_HDM_cars'] = n_cars - int(IDM_pct*n_cars)
    #
	# v = np.ones(n_cars) * params['init_v']# Initial velocities (in m/s)
	# x_vec = np.linspace(0,end_of_track-end_of_track/5,n_cars)	# Initial positions
	# # reverse initial positions
	# x_vec = x_vec[::-1]
	# ts = np.linspace(t_start,total_time,t_steps) # time steps
	# x_v_vec = np.concatenate(([x_vec], [v]), axis=0).reshape(1,-1)[0]
	# # Runge Kutta isn't quite working, but I haven't looked into why
	# y_s = []
	# y_s.append(x_v_vec)
	# #params['past_v_s'].append(v)
	# for i in range(1,len(ts)):
    #
	# 	# y_next = runge_kutta_blended_4(y_s[-1], x_v_dash, x_v_dash2, ts[i], ts[i]-ts[i-1],IDM_idx, HDM_idx, params, past)
	# 	# x_next = y_next[:n_cars]
	# 	# v_next = y_next[n_cars:]
	# 	# v_next[v_next<0]=0
	# 	# y_next = np.concatenate((x_next, v_next), axis=0)
	# 	# # y_next = runge_kutta_blended_5(y_s[-1], x_v_dash, x_v_dash2, ts[i], ts[i]-ts[i-1],IDM_idx, HDM_idx, params, past)
	# 	# # pdb.set_trace()
	# 	# y_s.append(y_next)
    #     # runge_kutta_4_driverparams(x_v_vec_k, x_v_dash, t_k, h, params, driveridx, driverparams):
	# 	y_s.append(runge_kutta_4_driverparams(y_s[-1], x_v_dash2, ts[i], ts[i]-ts[i-1],params,past, driveridx, driverparams))
	# 	#params['past_v_s'].append(params['past_y_s'][-1][n_cars:])
	# 	if i %200 == 0:
	# 		print "Finished Iteration: i={}".format(i)
	# y_s = np.array(y_s)
	# #y_s = sp.integrate.odeint(x_v_dash, y0=x_v_vec, t=ts,args=(params,))
    #
	# # Plot position and velocity of each car
	# fig, axes = plt.subplots(1,2, figsize=(16,8))
	# # Plot positions over time
	# for car in xrange(n_cars):
	# 	axes[0].plot(ts, y_s[:,car])
	# # Plot velocity over time
	# for car in xrange(n_cars):
	# 	axes[1].plot(ts, y_s[:,car+n_cars])
	# axes[0].set_xlabel('Time')
	# axes[0].set_ylabel('Position')
	# axes[1].set_xlabel('Time')
	# axes[1].set_ylabel('Velocity')
	# plot_out_name = '../figures/blended_{}.png'.format(params['n_a'])
	# plt.savefig(plot_out_name,
	# 			orientation='landscape',format='png',edgecolor='black')
