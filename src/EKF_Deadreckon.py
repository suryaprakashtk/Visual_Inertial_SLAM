import numpy as np
from utils import *


def dead_reckon_IMU(time,linear_velocity,angular_velocity):
    linear_velocity_var = 1
    ang_vel_var = 0.05
    mean_t_t = np.zeros(shape=(4,4,time.shape[1]),dtype=float)
    sigma_t_t = np.zeros(shape=(6,6,time.shape[1]),dtype=float)

    sigma_t_t[0,0,0] = linear_velocity_var
    sigma_t_t[1,1,0] = linear_velocity_var
    sigma_t_t[2,2,0] = linear_velocity_var
    sigma_t_t[3,3,0] = ang_vel_var
    sigma_t_t[4,4,0] = ang_vel_var
    sigma_t_t[5,5,0] = ang_vel_var

    tau = np.zeros(shape=(time.shape[1]-1,1,1),dtype=float)
    tau[:,0,0] = time[0,1:] - time[0,:-1]
    mean_t_t[:,:,0] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    
    u_t = np.concatenate((linear_velocity.T,angular_velocity.T),axis=1)
    u_t_hat = axangle2twist(u_t)
    
    tau_u_t_hat = u_t_hat[:-1,:,:]*tau
    exp_twist = twist2pose(tau_u_t_hat)
    
    for i in range(1,time.shape[1]):
        mean_t_t[:,:,i] = mean_t_t[:,:,i-1]@exp_twist[i-1,:,:]

    # uncomment for visualization
    visualize_trajectory_2d_dead_final(mean_t_t,show_ori=True)

    return mean_t_t
