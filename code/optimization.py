import intelligent_driver_model
import scipy as sp
import scipy.integrate
import pdb
import numpy as np

# define a wrapper function to determine what cost is being calculated
def cost_wrapper(optimize_params, cost_func, optimize_variables, params):
	for var, param in zip(optimize_variables, optimize_params):
		params[var] = param

	simulation_result = intelligent_driver_model.run_simulation(params)
	# pdb.set_trace()
	x_v_dash = np.array(params['x_v_dash'])

	displacement = simulation_result[:,:params['n_cars']].T
	velocity = simulation_result[:,params['n_cars']:].T
	acceleration = x_v_dash[:,params['n_cars']:].T

	# We should be able to write cost functions based on these three variables
	c = cost_func(displacement, velocity, acceleration, params)
	return c

# Define the cost functions:

# Cost Function 1
# Minimizing the total travel time -> Page 419
# Note: Minimmising the travel time is the same as maximising the
# displacement across the time interval
def maximise_displacement(displacement, velocity, acceleration, params):
	return 1-np.sum(displacement[:,-1])/float(params['n_cars'])

# Cost Function 2
# Maximizing the driving comfort (minimise discomfort) -> Page 419
def comfort_cost(displacement, velocity, acceleration, params):
	tau0 = 	1. #s -> characteristic time
	a0 = 		1. # m/s**2
	T0 = 		1. # arbitrary normalisation constant

	# enforce smooth boundary condition
	acceleration_dash = np.concatenate((np.zeros((params['n_cars'],1)), np.diff(acceleration)/params['t_step']), axis=1)
	integrand = np.sum(acceleration**2 + tau0**2*acceleration_dash**2, axis=0)
	cost = 1/(T0*a0**2)*sp.integrate.trapz(
									y=integrand,
									axis=0
								)
	return cost

def comfort_cost_nick(displacement, velocity, acceleration, params):
	cost = np.sum(np.abs(acceleration[:,-1]))/float(params['n_cars'])
	return cost

# Cost Function 3
# Maximise average velocity
def maximise_velocity(displacement, velocity, acceleration, params):
	return 1-np.sum(velocity[:,-1])/float(params['n_cars'])

# Cost Function 4
# Combination model
def combination_model(displacement, velocity, acceleration, params):
	return maximise_velocity(displacement, velocity, acceleration, params) + maximise_displacement(displacement, velocity, acceleration, params)/100. +comfort_cost_nick(displacement, velocity, acceleration, params)

if __name__ == '__main__':
	# define the params in this context
	params 									= dict()
	params['v0'] 						= 30.0 # desired velocity (in m/s) of vehicles in free traffic
	params['init_v'] 				= 5.0 # initial velocity
	params['T'] 						= 1.5 # Safe following time
	params['a'] 						= 1.0 # Maximum acceleration (in m/s^2)
	params['b'] 						= 2.0 # Comfortable deceleration (in m/s^2)
	params['delta'] 				= 4.0 # Acceleration exponent
	params['s0'] 						= 2. # minimum gap (in m)
	params['end_of_track'] 	= 600 # in m
	params['t_steps'] 			= 1000 # number of timesteps
	params['n_cars'] 				= 50 # number of vehicles
	params['total_time'] 		= 500 # total time (in s)
	params['c'] 						= 0.99 # correction factor
	params['t_start']  			= 0.0
	params['t_step'] = (params['total_time']-params['t_start'])/float(params['t_steps'])
	params['IDM_model_num'] = 0 # set the model to work with (0,1,2)
	params['x_v_dash'] 			= [np.zeros(2*params['n_cars'])]

	# This runs an optimisation over comfort
	# solution = sp.optimize.minimize(cost_wrapper, [5.] , args=(comfort_cost_nick, ['v0'], params), bounds=[(2., 100.)])

	# This runs an optimisation over distance travelled
	# solution = sp.optimize.minimize(cost_wrapper, [4, 10] , args=(maximise_displacement, ['T', 'v0'], params), bounds=[(1.5, 10), (5, 500)])

	# This runs an optimisation to get best average velocity
	# solution = sp.optimize.minimize(cost_wrapper, [2.5, 25] , args=(maximise_velocity, ['s0', 'v0'], params), bounds=[(2, 100), (5, 100)])
	# This runs an optimisation over distance travelled

	solution = sp.optimize.minimize(cost_wrapper, [5.] , args=(combination_model, ['v0'], params), bounds=[(2., 500.)])
	# solution = sp.optimize.minimize(cost_wrapper, [5.] , args=(maximise_displacement, ['v0'], params), bounds=[(2., 100.)])

	print solution.x
	pdb.set_trace()









