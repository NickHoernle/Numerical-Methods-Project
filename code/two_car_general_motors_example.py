import numpy as np
import matplotlib.pyplot as plt
from car_following_models import GeneralMotorsModel


def leadcar_problem1(t):
	"""
	Function that returns the lead car's acceleration as a 
	function of time.  In this example, the has a = 0 for the
	first 2 seconds, then has a = 1 for 2 seconds, then has
	a = -1 for 2 seconds, and then a = 0.
	"""
	# 
	if t >= 2 and t < 4:
		# accelerate by 1.0 m/s^2
		return 1.0
	elif t >= 4 and t <= 6:
		# accelerate by -1.0 m/s^2
		return -1.0
	else:
		# t < 2 or t > 6
		return 0.0

def main():
	'''
	Very basic example using General motor's car following model with 2 cars.
	The lead car has the following acceleration:
		- a=0 for first two seconds
		- a=1 for next two seconds
		- a=-1 for next two seconds
	Plots of position, velocity, and acceleration are created and stored in
	the figures directory.
	'''

	# set number of cars
	n = 2
	# set initial pos, velocity, accel
	x0 = np.array([0.0,28.0])
	v0 = np.array([16.0,16.0])
	a0 = np.array([0.0,0.0])
	# set lead car acceleration function
	lead_car_accel = leadcar_problem1
	# create model
	model = GeneralMotorsModel(n,x0,v0,a0,lead_car_accel)
	# run simulation
	model.run_sim()
	# plot a,x,v of the cars
	model.plot('v')
	model.plot('x')
	model.plot('a')


if __name__ == '__main__':
	main()