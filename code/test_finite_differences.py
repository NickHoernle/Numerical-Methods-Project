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
from intelligent_driver_model import *
# from human_driver_model import *
from collections import deque

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
    params['n_cars'] 				= 50 # number of vehicles
    params['total_time'] 		= 500 # total time (in s)
    params['c'] 						= 0.99 # correction factor
    params['t_start'] = 0.0

    params['x_v_dash'] 			= [2*np.zeros(params['n_cars'])]

    v0 = params['v0']
    init_v = params['init_v']
    T = params['T']
    a = params['a']
    b = params['b']
    delta = params['delta']
    s0 = params['s0']
    end_of_track = params['end_of_track']
    n_cars = params['n_cars']
    total_time = params['total_time']
    for i in range(200,300,10):
        params['t_steps'] 			= i # number of timesteps
        params['t_step'] = (params['total_time']-params['t_start'])/float(params['t_steps'])
        t_steps = params['t_steps']
        params['IDM_model_num'] = 1

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
            # y_s.append(fwd_euler(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
            # y_s.append(heuns_method(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
            # y_s.append(runge_kutta_4(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
            # y_s.append(runge_kutta_5(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
        	y_s.append(runge_kutta_3(y_s[-1], x_v_dash, ts[i], ts[i]-ts[i-1], params))
        y_s = np.array(y_s)
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
        plot_out_name = "../figures/displacement_and_velocity_plot_tsteps{}.png".format(params['t_steps'])
        plt.savefig(plot_out_name,
        		orientation='landscape',format='png',edgecolor='black')
        # plt.close()
        plt.show()
