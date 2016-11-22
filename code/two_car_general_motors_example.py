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
		return 4.0
	elif t >= 4 and t <= 6:
		# accelerate by -1.0 m/s^2
		return -4.0
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
	ns = [2,3,10,40] # number of cars
	max_ts = [20.0,20.0,100.0,100.0] # to see effect for different number of cars
	for j in xrange(len(ns)):
		n = ns[j]
		max_t = max_ts[j]
		# set initial pos, velocity, accel
		x0 = np.array([28.0*i for i in xrange(n)])
		v0 = np.ones(n)*16.0
		a0 = np.zeros(n)
		
		# set lead car acceleration function
		lead_car_accel = leadcar_problem1
		# create model
		model = GeneralMotorsModel(n,x0,v0,a0,lead_car_accel,max_t=max_t)
		# run simulation
		model.run_sim()
		# plot a,x,v of the cars
		model.plot('v')
		model.plot('x')
		model.plot('a')


if __name__ == '__main__':
	main()