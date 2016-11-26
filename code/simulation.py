"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import pdb


n_cars = 5 # use later
meters = 100000
v_target = 26.8 # 60miles/h in m/s
d_target = 5 # ideal spacing of 5m between cars
# velocity = v_target

# timestep = 1.
# distance = 0
# gain = -2 # strength of controller
# acceleration = 0
# time = 0
# distances = []
# velocities = []
# times = []

# while distance < meters:
# 	distances.append(distance)
# 	velocities.append(velocity-v_target)
# 	times.append(time)

# 	time += timestep
# 	distance += velocity*timestep + 0.5*acceleration*timestep**2
# 	velocity = velocity + acceleration*timestep
# 	# try and control for the acceleration by the velocity readings
# 	acceleration = gain*(velocity - v_target)/timestep
# 	# add some noise to acceleration
# 	acceleration = acceleration + np.random.normal(0,0.1)
#!/usr/bin/python

#########################################
# Rycroft model

# Constants in model
# alpha=1.5
# beta=0.5
# gamma=0.4
# delta=0.4

# Function
# def deriv(x,t):
#   return np.array([alpha*x[0]-beta*x[0]*x[1],-gamma*x[1]+delta*x[0]*x[1]])

# Solve ODE using the "odeint" library in SciPy
# time=np.linspace(0,70,500)

# # Initial conditions, set to the initial
# xinit=np.array([10,5])
# x=odeint(deriv,xinit,time)

# for i in range(0,500):
#     print time[i],x[i,0],x[i,1]

# # Plot the solutions
# plt.figure()
# p0,=plt.plot(time,x[:,0])
# p1,=plt.plot(time,x[:,1])
# plt.legend([p0,p1],["B(t)","C(t)"])
# plt.xlabel('t')
# plt.ylabel('Population')
# plt.show()
########################################

velocity = np.ones(n_cars) * v_target
distance = np.arange(n_cars)*5 # everyone starts 5m apart

delta_t = 1.
alpha, beta = 0.001,0.001

def f(x,t):
	x = x.reshape(2,-2).T
	displacement = x[:,0]
	velocity = x[:,1]
	acceleration = np.zeros(len(displacement))

	for i,x in enumerate(displacement):
		delta_d = np.random.normal(0,0.01) # little noise
		acceleration[i] = 0.1*(v_target-velocity[i]) + delta_d
		if i != 0:
			delta_d = (displacement[i] - displacement[i-1]) #distance to car ahead
			delta_v = v_target-velocity[i] # is this dangerous???
			acceleration[i] = (alpha*(delta_d-d_target)/float(delta_d) + beta*delta_v)
			print acceleration[i]

	velocity = velocity + acceleration*delta_t
	displacement = displacement + velocity*delta_t

	return np.concatenate((displacement, velocity))

n_min = 60*60.
time=np.linspace(0,n_min-1,n_min) # one hour simulation
x = np.concatenate((distance, velocity))

x_vals = []
for i in time:
	x = f(x,delta_t)
	x_vals.append( x[5:] )

x_vals = np.array(x_vals)

for x in x_vals.T:
	print x
	plt.plot(time, x)
plt.show()
# while min(distance) < meters:
# 	for car in cars:

# plt.plot(times, velocities)
# plt.show()
# n_cars = 5 #cars
# t_steps = 1000
# u0 = 10*1000/(60.*60.) # 20km/h in m/s
# a0 = 0
# mu = 1e-4
# h = 1.

# position = np.zeros((n_cars,t_steps)).T
# position[0,:]=range(0,n_cars)
# velocity = np.zeros((n_cars,t_steps)).T
# velocity[0,:] = u0
# acceleration = np.zeros((n_cars,t_steps)).T
# acceleration[0,:] = a0

# fx = np.vectorize(lambda t,x0,u,a: x0 + u*t + 0.5*a*t**2)

# # at every time step we base the acceleration on the change in
# # distance to the car ahead. We then add some small random noise to
# # the acceleration to simulate the cars:
# for step, time in enumerate(np.linspace(0,t_steps-1,t_steps)):
# 	if step != 0:
# 		for car, pos in enumerate(position[step,:]):
# 			position[step,car] = fx(time, position[step-1,car], velocity[step-1,car], acceleration[step-1,car])
# 			a = 0
# 			if car != 0:
# 				# acceleration is directly proportional to the change in distance
# 				delta_d = (position[step-1,car-1]-position[step-1,car] - (position[step,car-1]-position[step,car]))/float(h)
# 				a = np.max([-1., delta_d/float(position[step,car-1]-position[step,car])])
# 				a = np.min([1, a])
# 			a = a + np.random.normal(0,1e-5)
# 			acceleration[step,car] = mu*a
# 			velocity[step,car] = 0.5*20 + 0.5*(acceleration[step,car]*time + velocity[step-1,car])

# fig, ax = plt.subplots()

# cars = ax.scatter(position[0, :], [1]*n_cars, c='g')
# cars.set_array(np.arange(n_cars))

# def animate(i):
# 	i = int(i)
# 	pos = position[i, :]
# 	new_pos = np.concatenate(([pos], [[1]*n_cars]), axis=0)
# 	cars.set_offsets(new_pos.T)
# 	ax.set_xlim([np.max(pos)-50, np.max(pos)])
# 	return cars,

# # Init only required for blitting to give a clean slate.
# def init():
# 	# cars.set_offsets(np.ma.array(np.array([position[0, :]], [[1]*n_cars]).T, mask=True))
# 	return cars,

# ani = animation.FuncAnimation(fig, animate, np.arange(10000), init_func=init, interval=25, blit=False)

# plt.show()





# race_track = lambda x,y: x**2/20. + y**2/20. # radius of 20 km
# fx = np.vectorize(lambda t: 20*np.sin(t))
# fy = np.vectorize(lambda t: 20.*np.cos(t))


# fig, ax = plt.subplots()

# t = np.linspace(0, 2*np.pi, 1000)
# t_pos = np.linspace(0, 2*np.pi, 30)

# line, = ax.plot(fx(t), fy(t))
# cars = ax.scatter(fx(t_pos), fy(t_pos), c='g')

# def animate(i):
# 	new_pos = np.concatenate(([fx(t_pos+i)], [fy(t_pos+i)]), axis=0)
# 	# cars.set_array(fy(t_pos + i))
# 	cars.set_offsets(new_pos.T)
# 	return cars,

# # Init only required for blitting to give a clean slate.
# def init():
# 	cars.set_offsets(np.ma.array([t_pos,t_pos], mask=True))
# 	return cars,

# ani = animation.FuncAnimation(fig, animate, np.linspace(1, 6, 200), init_func=init, interval=25, blit=False)

# plt.show()