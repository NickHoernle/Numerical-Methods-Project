import numpy as np
import matplotlib.pyplot as plt

class GeneralMotorsModel(object):
	'''
	Implementation of General Motor's car following model
	'''
	def __init__(self,n,x0,v0,a0,lead_car_accel,dT=1.0,
				dt=0.5,max_t=20.0,alpha=13.0,m=0.0,l=1.0):
		'''
		Parameters:
			- n: number of cars in system
			- x0: vector of initial positions (car i leads, car i-1 follows) (in meters)
			- v0: vector of initial velocities (in meters/sec)
			- a0: vector of initial accelerations (in meters/sec)
			- lead_car_accel: lead car's acceleration as a function of time
			- dT: reaction time (in seconds)
			- dt: update_time (in seconds)
			- max_t: end time for simulation (in seconds)
			- alpha: sensitivity coefficient
			- m: speed exponent (can take values -2 to +2)
			- l: distance headway exponent (can take values -1 to +4)
		'''
		self.n = n
		self.max_t = max_t
		self.dt=dt
		self.dT=dT
		self.alpha = alpha
		self.m = m
		self.l = l
		self.lead_car_accel=lead_car_accel

		# create array of update times 
		self.ts = np.arange(0.0,self.max_t+self.dt,self.dt)
		
		# Create matrices for position, velocity, accel
		# rows are timesteps, columns are cars
		# entry at x[i,j] is position of car j at timestep index i
		self.x = np.zeros((len(self.ts),self.n))
		self.v = np.zeros((len(self.ts),self.n))
		self.a = np.zeros((len(self.ts),self.n))
		self.x[0] = x0
		self.v[0] = v0
		self.a[0] = a0
		# index of timestep into position, velocity, accel arrays
		# note each timestep dt
		self.t_idx = 1
		# time
		self.t = dt
		# compute the reaction time in terms of timesteps
		self.dT_offset = dT/dt

	def run_sim(self):
		# run a simulation
		while self.t <= self.max_t:
			# compute velocities at next timestep
			self.v[self.t_idx] = self.v[self.t_idx-1] + self.a[self.t_idx-1]*self.dt
			# compute position at next timestep
			self.x[self.t_idx] = self.x[self.t_idx-1] + self.v[self.t_idx-1]*self.dt + 0.5*self.a[self.t_idx-1]*self.dt**2
			# check if t> reaction time
			reacted_to_time = int(self.t_idx-self.dT_offset)
			if reacted_to_time >= 0:
				# compute acceleration at next timestep for all cars except
				# except lead car
				reaction_dx = self.x[reacted_to_time,1:]-self.x[reacted_to_time,:-1]
				reaction_dv = self.v[reacted_to_time,1:]-self.v[reacted_to_time,:-1]
				self.a[self.t_idx,:-1] = (self.alpha * (self.v[reacted_to_time,:-1])**self.m)/(reaction_dx**self.l)*reaction_dv
			else:
				# if t < reaction time, the following car does not change
				self.a[self.t_idx,:-1] = self.a[self.t_idx-1,:-1]
			self.a[self.t_idx,-1] = self.lead_car_accel(self.t)
			self.t_idx += 1
			self.t += self.dt

	def plot(self,property):
		# plot property of each car as a function of time
		# property = 'v','x','a'

		if property == 'v':
			y = self.v
			ylabel = '$v$'
		elif property == 'a':
			y = self.a
			ylabel = '$a$'
		elif property == 'x':
			y = self.x
			ylabel = '$x$'
		plt.plot(self.ts,y)
		plt.xlabel('$t$')
		plt.ylabel(ylabel)
		out_name = '../figures/general_motors_model_{}car_{}.pdf'.format(self.n,property)
		plt.savefig(out_name,orientation='landscape',format='pdf',edgecolor='black')
		plt.clf()
