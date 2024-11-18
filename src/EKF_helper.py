import numpy as np
import scipy
from utils import *

def init_motion_model(no_of_timesteps,linear_velocity_noise,angular_velocity_noise):
    mean = np.zeros(shape=(4,4,no_of_timesteps),dtype=float)
    sigma = np.zeros(shape=(6,6,no_of_timesteps),dtype=float)
    mean[:,:,0] = np.eye(4)
    sigma[:,:,0] = np.eye(6)
    sigma[0:3,0:3,0] = linear_velocity_noise * sigma[0:3,0:3,0]
    sigma[3:6,3:6,0] = angular_velocity_noise * sigma[3:6,3:6,0]
    return [mean,sigma]

def init_landmark(features,pose,timestep,cov_init,imu_T_cam,K,b,mean = np.eye(3),covariance = np.eye(3),mask = np.eye(3)):
    if(timestep == 0):
        mean = np.zeros(shape=(3,features.shape[1]),dtype=float)
        mean[:,:] = -1
        covariance = np.zeros(shape=(3*features.shape[1],3*features.shape[1]),dtype=float)
        mask = np.any(features[:,:,0] != -1, axis=0)
    

    mean[:,mask] = obs_z_to_m_mat(features[:,mask,timestep],pose,imu_T_cam,K,b)
    indValid = np.argwhere(mask==True)
    for i in range(indValid.shape[0]):
        covariance[3*indValid[i,0]:3*indValid[i,0]+3,3*indValid[i,0]:3*indValid[i,0]+3] = cov_init*np.eye(3)
    
    return [mean,covariance]

def create_mask(features,mean):
    visible_features = np.any(features != -1, axis=0)
    prior_not_initialized = np.any(mean[:,:] == -1, axis =0)
    prior_initialized = np.any(mean[:,:] != -1, axis =0)
    ind2initialize = np.logical_and(visible_features,prior_not_initialized)
    EKF_update_ind = np.logical_and(visible_features,prior_initialized)
    return [ind2initialize,EKF_update_ind]

def prediction_motion_model(time,linear_velocity,angular_velocity,prior_mean,prior_sigma,v_noise,w_noise):
    W = np.eye(6)
    W[0,0] = v_noise
    W[1,1] = v_noise
    W[2,2] = v_noise
    W[3,3] = w_noise
    W[4,4] = w_noise
    W[5,5] = w_noise

    u_t = np.concatenate((linear_velocity,angular_velocity),axis=0)
    u_t_hat = axangle2twist(u_t)
    tau_u_t_hat = u_t_hat*time
    exp_twist = scipy.linalg.expm(tau_u_t_hat)
    predicted_mean = prior_mean@exp_twist

    u_t_curl_hat = axangle2adtwist(u_t)
    tau_u_t_curl_hat = -u_t_curl_hat*time
    exp_ad_twist = scipy.linalg.expm(tau_u_t_curl_hat)
    # sigma_rr = prior_sigma[-6:,-6:]
    predicted_sigma = exp_ad_twist@prior_sigma@exp_ad_twist.T + W
    # crpss correlation
    # prior_sigma[-6:,-6:] = predicted_sigma_rr
    # prior_sigma[:-6,-6:] = prior_sigma[:-6,-6:]@exp_ad_twist.T
    # prior_sigma[-6:,:-6] = exp_ad_twist@prior_sigma[-6:,:-6]
    sigma_eig = np.linalg.eigvals(predicted_sigma)
    is_psd = np.all(sigma_eig < 0)
    if(is_psd):
        print("pose covariance is not PSD")

    return [predicted_mean,predicted_sigma]

def obs_z_to_m_mat(Z_feature,world_T_body,imu_T_cam,K,b):
    disparity = Z_feature[0,:] - Z_feature[2,:]
    fsub = K[0,0] * b
    z = fsub/disparity
    pixels_u = Z_feature[0,:]
    pixels_v = Z_feature[1,:]
    last_row = np.ones(shape=(disparity.shape[0]),dtype=float)
    local_pixels = np.vstack([pixels_u,pixels_v,last_row])
    k_inv = np.linalg.inv(K)
    temp = k_inv@local_pixels
    x = temp[0,:]*z
    y = temp[1,:]*z
    z = temp[2,:]*z
    last_row = np.ones(shape=(disparity.shape[0]),dtype=float)
    local_pts = np.vstack([x,y,z,last_row])
    M_world = world_T_body[:,:]@imu_T_cam@local_pts

    return M_world[0:3,:]

def obs_m_to_z_mat(M_world,world_T_body,imu_T_cam,K,b):

    cam_T_imu = np.linalg.inv(imu_T_cam)
    body_T_world = np.linalg.inv(world_T_body)
    last_row = np.ones(shape=(M_world.shape[1]),dtype=float)
    mu_t_homo = np.vstack([M_world,last_row])

    xyz = cam_T_imu@body_T_world@mu_t_homo
    xyz_projected_T = projection(xyz.T)
    xyz_projected = xyz_projected_T.T
    ks = np.array([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[K[0,0],0,K[0,2],-K[0,0]*b],[0,K[1,1],K[1,2],0]])
    Z_feature = ks@xyz_projected
    return Z_feature

def innovation_land(prior_mean,features,predicted_pose,imu_T_cam,K,b,update_ind_mask,timestep):
    prior_mat = prior_mean[:,update_ind_mask]
    prior = prior_mat.reshape(-1,order='F')

    z_t_plus_1_mat = features[:,update_ind_mask]
    z_t_plus_1 = z_t_plus_1_mat.reshape(-1,order='F')

    z_tilda_mat = obs_m_to_z_mat(prior_mat,predicted_pose,imu_T_cam,K,b)
    z_tilda = z_tilda_mat.reshape(-1,order='F')

    innovation = z_t_plus_1 - z_tilda

    if(np.max(innovation)>10):
        print("Max innovvation in land " + str(timestep) + " = " + str(np.max(innovation)))
    if(np.min(innovation)<-10):
        print("Min innovvation in land " + str(timestep) + " = " + str(np.min(innovation)))
    return innovation

def gain_land(H,Sigma,V,timestep):
    
    v_noise = V*np.eye(H.shape[0])
    k_gain_T = np.linalg.solve(H@Sigma@H.T + v_noise,H@Sigma)
    k_gain = k_gain_T.T
    abc = np.linalg.eigvals(H@Sigma@H.T + v_noise)
    is_not_psd = np.all(abc <= 0)
    if(is_not_psd):
        print("HSigmaHT Land plus noise is not PD" + " at step " + str(timestep))
    if(np.max(k_gain)>10):
        print("Max Gain in land " + str(timestep) + " = " + str(np.max(k_gain)))
    if(np.min(k_gain)<-10):
        print("Min Gain in land " + str(timestep) + " = " + str(np.min(k_gain)))
    
    return k_gain

def gain_comb(H,Sigma,V,timestep):
    
    v_noise = V*np.eye(H.shape[0])
    k_gain_T = np.linalg.solve(H@Sigma@H.T + v_noise,H@Sigma)
    k_gain = k_gain_T.T
    abc = np.linalg.eigvals(H@Sigma@H.T + v_noise)
    is_not_psd = np.all(abc <= 0)
    if(is_not_psd):
        print("HSigmaHT comb plus noise is not PD" + " at step " + str(timestep))
    if(np.max(k_gain)>10):
        print("Max Gain in comb " + str(timestep) + " = " + str(np.max(k_gain)))
    if(np.min(k_gain)<-10):
        print("Min Gain in comb " + str(timestep) + " = " + str(np.min(k_gain)))
    
    return k_gain

def gain_robo(H,Sigma,v_noise,timestep):
    V = v_noise*np.eye(H.shape[0])
    k_gain_T = np.linalg.solve(H@Sigma@H.T + V,H@Sigma)
    k_gain = k_gain_T.T
    abc = np.linalg.eigvals(H@Sigma@H.T + V)
    is_not_psd = np.all(abc <= 0)
    if(is_not_psd):
        print("HSigmaHT Robo plus noise is not PD" + " at step " + str(timestep))
    
    if(np.max(k_gain)>10):
        print("Max Gain in robo " + str(timestep) + " = " + str(np.max(k_gain)))
    if(np.min(k_gain)<-10):
        print("Min Gain in robo " + str(timestep) + " = " + str(np.min(k_gain)))
    return k_gain

def update_land(k_gain,innovation,H,update_ind_mask,prior_mean,prior_sigma,timestep):
    EKF_update_positions = np.argwhere(update_ind_mask==True)
    subindex = block_submatrix_covariance(EKF_update_positions[:,0])
    mean_correction = k_gain @ innovation
    prior_mat = prior_mean[:,update_ind_mask]
    prior = prior_mat.reshape(-1,order='F')
    prior_new = prior + mean_correction[subindex]
    prior_mat_new = prior_new.reshape((3,-1), order = 'F')

    prior_sigma_new = prior_sigma - k_gain@H@prior_sigma
    temp2 = prior_sigma - prior_sigma.T
    if (np.max(temp2)>0.5):
        print("old Cov is not symmeteric in land update " + str(timestep))
    if(np.min(temp2)<-0.5):
        print("old Cov is not symmeteric in land update " + str(timestep))
    temp = prior_sigma_new - prior_sigma_new.T
    if(np.max(temp)>0.5):
        print("new Cov is not symmeteric in land update " + str(timestep))
    if(np.min(temp)<-0.5):
        print("new Cov is not symmeteric in land update " + str(timestep))
    return [prior_mat_new,prior_sigma_new]



def update_robo(k_gain,innovation,H,update_ind_mask,prior_mean,prior_sigma,timestep):

    error_twist = k_gain @ innovation

    error_hat = axangle2twist(error_twist)
    exp_twist = scipy.linalg.expm(error_hat)
    updated_mean = prior_mean@exp_twist

    term1_cov = np.eye(6) - k_gain@H
    updated_sigma = prior_sigma - k_gain@H@prior_sigma

    temp2 = prior_sigma - prior_sigma.T
    if (np.max(temp2)>0.5):
        print("old Cov is not symmeteric in robo update " + str(timestep))
    if(np.min(temp2)<-0.5):
        print("old Cov is not symmeteric in robo update " + str(timestep))
    temp = updated_sigma - updated_sigma.T
    if(np.max(temp)>0.5):
        print("new Cov is not symmeteric in robo update " + str(timestep))
    if(np.min(temp)<-0.5):
        print("new Cov is not symmeteric in robo update " + str(timestep))
    return [updated_mean,updated_sigma]

def H_matrix_calc_land(prior_mean,world_T_body,imu_T_cam,K,b,update_ind_mask):
    mu_t = prior_mean[:,update_ind_mask]
    H_update_positions = np.argwhere(update_ind_mask==True)
    output_h = np.zeros(shape=(4,3,mu_t.shape[1]),dtype=float)
    cam_T_imu = np.linalg.inv(imu_T_cam)
    body_T_world = np.linalg.inv(world_T_body)
    last_row = np.ones(shape=(mu_t.shape[1]),dtype=float)
    mu_t_homo = np.vstack([mu_t,last_row])
    xyz = cam_T_imu@body_T_world@mu_t_homo
    # This is correct
    xyz_jacob_projected_T = projectionJacobian(xyz.T)

    p = np.zeros(shape=(3,4),dtype=float)
    p[0:3,0:3] = np.eye(3)
    right_term = cam_T_imu@body_T_world@p.T
    ks = np.array([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[K[0,0],0,K[0,2],-K[0,0]*b],[0,K[1,1],K[1,2],0]])
    final_h = np.zeros(shape=(4*mu_t.shape[1],3*update_ind_mask.shape[0]),dtype=float)
    for i in range(mu_t.shape[1]):
        final_h[4*i:4*i+4,3*H_update_positions[i,0]:3*H_update_positions[i,0]+3] = ks@xyz_jacob_projected_T[i,:,:]@right_term


    return final_h

def circle_dot(input):
    output = np.zeros(shape=(4,6),dtype=float)
    output[0:3,0:3] = np.eye(3)
    hat = axangle2skew(input[0:3])
    output[0:3,3:6] = -1*hat
    return output

def H_matrix_calc_robo(prior_mean,world_T_body,imu_T_cam,K,b,update_ind_mask):
    mu_t = prior_mean[:,update_ind_mask]

    output_h = np.zeros(shape=(4,6,mu_t.shape[1]),dtype=float)
    final_h = np.zeros(shape=(4*mu_t.shape[1],6),dtype=float)
    ks = np.array([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[K[0,0],0,K[0,2],-K[0,0]*b],[0,K[1,1],K[1,2],0]])

    cam_T_imu = np.linalg.inv(imu_T_cam)
    body_T_world = np.linalg.inv(world_T_body)
    last_row = np.ones(shape=(mu_t.shape[1]),dtype=float)
    mu_t_homo = np.vstack([mu_t,last_row])
    xyz = cam_T_imu@body_T_world@mu_t_homo
    xyz_jacob_projected_T = projectionJacobian(xyz.T)

    xyz_body = body_T_world@mu_t_homo
    for i in range(mu_t.shape[1]):
        term1 = circle_dot(xyz_body[:,i])
        final_h[4*i:4*i+4,:] = -ks@xyz_jacob_projected_T[i,:,:]@cam_T_imu@term1

    return final_h


def block_submatrix_covariance(feature_index):
    # given feature number, it gives the row/columne number of the matrix to check
    feature_index = 3*feature_index
    repeat_times = 3
    output_matrix = feature_index + np.arange(repeat_times).reshape(-1, 1)
    output = output_matrix.reshape(-1,order='F')
    return output


def assemble(H_land,H_robo,sigma_land,sigma_robo,timestep):
    h_output = np.hstack([H_land,H_robo])
    sigma_output = np.zeros(shape=(sigma_land.shape[0]+6,sigma_land.shape[0]+6))
    sigma_output[:-6,:-6] = sigma_land
    sigma_output[-6:,-6:] = sigma_robo
    temp = sigma_output - sigma_output.T
    if(np.max(temp)>0.5):
        print("new Cov is not symmeteric in land update " + str(timestep))
    if(np.min(temp)<-0.5):
        print("new Cov is not symmeteric in land update " + str(timestep))

    return [h_output,sigma_output]

def deassemble(K_comb,timestep):
    K_land = K_comb[:-6,:]
    K_robo = K_comb[-6:,:]
    return [K_land,K_robo]