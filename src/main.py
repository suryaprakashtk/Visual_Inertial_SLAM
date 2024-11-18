import numpy as np
from utils import *
from EKF_Deadreckon import *
from EKF_Mapping import *
from EKF_SLAM import *

if __name__ == '__main__':
	path = "./"
	# Load the measurements
	filename = path + "data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	time = t - t[0,0]
	features = features[:,::20,:]

	# (a) IMU Localization via EKF Prediction
	dead_pose = dead_reckon_IMU(time,linear_velocity,angular_velocity)
	# (b) Landmark Mapping via EKF Update
	EKF_Mapping(time,linear_velocity,angular_velocity,features,K,b,imu_T_cam)
	# (c) Visual-Inertial SLAM
	VI_Slam(time,linear_velocity,angular_velocity,features,K,b,imu_T_cam)
	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


