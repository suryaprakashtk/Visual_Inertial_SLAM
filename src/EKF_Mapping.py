import numpy as np
from utils import *
from EKF_helper import *

def dead_reckon_map(time,dead_pose,features,K,b,imu_T_cam):
    # scaled_ind = np.arange(0,features.shape[1],10)
    # features = features[:,:,:]
    landmarks_init = np.zeros(shape=(4,features.shape[1]),dtype=float)
    landmarks_init[:,:] = -1

    global_pts = np.zeros(shape=(4,1),dtype=float)
    for i in range(features.shape[2]):
        mask = np.any(features[:,:,i] != -1, axis=0)
        mask_not_initialized = np.any(landmarks_init[:,:] == -1, axis =0)
        indValid = np.logical_and(mask,mask_not_initialized)
        landmarks_init[:,indValid] = features[:,indValid,i]

        disparity = features[0,indValid,i] - features[2,indValid,i]
        fsub = K[0,0] * b
        z = fsub/disparity
        pixels_u = features[0,indValid,i]
        pixels_v = features[1,indValid,i]
        last_row = np.ones(shape=(disparity.shape[0]),dtype=float)
        local_pixels = np.vstack([pixels_u,pixels_v,last_row])
        k_inv = np.linalg.inv(K)
        temp = k_inv@local_pixels
        
        x = temp[0,:]*z
        y = temp[1,:]*z
        z = temp[2,:]*z
        last_row = np.ones(shape=(disparity.shape[0]),dtype=float)
        local_pts = np.vstack([x,y,z,last_row])
        temp = dead_pose[:,:,i]@imu_T_cam@local_pts
        
        global_pts = np.concatenate((global_pts, temp), axis=1)


    visualize_trajectory_2d_dead(dead_pose,show_ori=True,m=global_pts)
    return



def EKF_Mapping(time,linear_velocity,angular_velocity,features,K,b,imu_T_cam):

    imu_T_cam = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])@imu_T_cam
    lin_vel_noise = 1e-10
    ang_vel_noise = 1e-10
    cov_init_land = 2
    motion_noise_v = 1e-10
    motion_noise_w = 1e-10
    obs_noise_v = 2
    [mean_robo_prior,sigma_robo_prior] = init_motion_model(time.shape[1],lin_vel_noise,ang_vel_noise)
    [mean_robo_predict,sigma_robo_predict] = init_motion_model(time.shape[1],lin_vel_noise,ang_vel_noise)
    [mean_land_prior,sigma_land_prior] = init_landmark(features,mean_robo_prior[:,:,0],0,cov_init_land,imu_T_cam,K,b)
    new_features_observed = []
    for i in range(1,time.shape[1]):
        tau = time[0,i] - time[0,i-1] 
        [mean_robo_predict[:,:,i],sigma_robo_predict[:,:,i]] = prediction_motion_model(tau,linear_velocity[:,i],angular_velocity[:,i],mean_robo_prior[:,:,i-1],sigma_robo_prior[:,:,i-1],motion_noise_v,motion_noise_w)
        
        [ind2initialize,EKF_update_ind] = create_mask(features[:,:,i],mean_land_prior)
        init_positions = np.argwhere(ind2initialize==True)
        if(init_positions.shape[0]!=0):
            [mean_land_prior,sigma_land_prior] = init_landmark(features,mean_robo_predict[:,:,i],i,cov_init_land,imu_T_cam,K,b,mean = mean_land_prior,covariance=sigma_land_prior,mask = ind2initialize)
            new_features_observed.append(i)
            
        EKF_update_positions = np.argwhere(EKF_update_ind==True)
        if(EKF_update_positions.shape[0]==0):
            print("No EKF step at step " + str(i))
            mean_robo_prior[:,:,i] = mean_robo_predict[:,:,i]
            sigma_robo_prior[:,:,i] = sigma_robo_predict[:,:,i]
            continue

        inno_land = innovation_land(mean_land_prior,features[:,:,i],mean_robo_predict[:,:,i],imu_T_cam,K,b,EKF_update_ind,i)

        H_land = H_matrix_calc_land(mean_land_prior,mean_robo_predict[:,:,i],imu_T_cam,K,b,EKF_update_ind)
        H_robo = H_matrix_calc_robo(mean_land_prior,mean_robo_predict[:,:,i],imu_T_cam,K,b,EKF_update_ind)

        [H_comb,sigma_comb] = assemble(H_land,H_robo,sigma_land_prior,sigma_robo_predict[:,:,i],i)
        K_comb = gain_comb(H_comb,sigma_comb,obs_noise_v,i)
        [K_land,K_robo] = deassemble(K_comb,i)

        [mean_land_prior[:,EKF_update_ind],sigma_land_prior] = update_land(K_land,inno_land,H_land,EKF_update_ind,mean_land_prior,sigma_land_prior,i)
        # [mean_robo_prior[:,:,i],sigma_robo_prior[:,:,i]] = update_robo(K_robo,inno_land,H_robo,EKF_update_ind,mean_robo_predict[:,:,i],sigma_robo_predict[:,:,i],i)
        mean_robo_prior[:,:,i] = mean_robo_predict[:,:,i]
        sigma_robo_prior[:,:,i] = sigma_robo_predict[:,:,i]
        
        print(i)
        if(i%1000==0):
            plot_mask_land = np.any(mean_land_prior[:,:] != -1, axis =0)
            plot_mean_land = mean_land_prior[:,plot_mask_land]
            visualize_trajectory_map_final(mean_robo_prior[:,:,:i],show_ori=False,m=plot_mean_land,iter_no = i)
    visualize_trajectory_map_final(mean_robo_prior[:,:,:i],show_ori=False,m=plot_mean_land,iter_no = i)
    return