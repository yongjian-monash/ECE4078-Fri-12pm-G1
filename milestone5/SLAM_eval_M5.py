# evaluate the map generated by SLAM against the true map
import ast
import numpy as np
import json
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF

def parse_groundtruth(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline()) 
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            usr_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                usr_dict = ast.literal_eval(f.readline()) 
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict
    
def parse_generated_gt(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline()) 
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def match_aruco_points(aruco0 : dict):
    points0 = []
    keys = []
    aruco1=[1,2,3,4,5,6,7,8,9,10]
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        keys.append(key)
    return keys, np.hstack(points0)

def match_aruco_points_test(aruco0 : dict, aruco1 : dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
    return keys, np.hstack(points0), np.hstack(points1)

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])


    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1/num_points * np.reshape(np.sum(points1, axis=1),(2,-1))
    mu2 = 1/num_points * np.reshape(np.sum(points2, axis=1),(2,-1))
    sig1sq = 1/num_points * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/num_points * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1,0],R[0,0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    # print(R)
    # print(points)
    # print(x)
    # print(R@points)

    points_transformed =  R @ points + x
    return points_transformed
    
def solve_trans_rot(points1, points2):
    # points_1 = robot_cur (Real)
    # points_2 = robot_ekf (SLAM)

    #print(points1)
    print(points2[2])
    theta =  points2[2][0] - points1[2]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    x = np.array([[points2[0][0]],[points2[1][0]]]) - R @ np.array([[points1[0]],[points1[1]]])
    return theta, x

def apply_transform_custom(theta, x, points, offset_rot = 0, offset_x = 0, offset_y = 0):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    theta += offset_rot
    x[0,0] += offset_x
    x[1,0] += offset_y
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    # print(R)
    # print(points)
    # print(x)
    # print(R@points)

    points_transformed =  R @ points + x
    return points_transformed

def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)

def givecoord(robot_pose):

    us_aruco = parse_user_map('lab_output/slam.txt')
    taglist, us_vec= match_aruco_points(us_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec = us_vec[:,idx]
    x = -np.array([[robot_pose[0][0]],[robot_pose[1][0]]])
    theta = -robot_pose[2][0]
    us_vec_aligned = apply_transform_custom(theta, x, us_vec)

    #exporting coordinates to txt file
    px, py = [], []
    px = us_vec_aligned[0]
    py = us_vec_aligned[1]

    d = {}
    for i in range(len(px)):
        d['aruco' + str(i+1) + '_0'] = {'x': px[i], 'y':py[i]}
        
    with open('testing_chris.txt', 'w') as f:
        print(d, file=f)

    print()
    print("The following parameters optimally transform the estimated points to the ground truth.")
    print("Rotation Angle: {}".format(theta))
    print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    print()
    print("Number of found markers: {}".format(len(taglist)))
    
    print()
    print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    print('-----------------------------------------------------------------')
    for i in range(len(taglist)):
        print('%3d %9.2f %9.2f\n' % (taglist[i], us_vec_aligned[0][i], us_vec_aligned[1][i]))
    
    ax = plt.gca()
    ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
    for i in range(len(taglist)):
        ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Pred'])
    plt.grid()
    plt.show()
    plt.close()
    plt.savefig('pics/test_plot_markers.png')

def save(markers, fname="slam_aligned.txt"):
    # base_dir = Path('./')
    d = {}
    for i in range(10):
        d['aruco' + str(i+1) + '_0'] = {'x': round(markers[0][i], 1), 'y':round(markers[1][i], 1)}
    map_attributes = d
    with open(fname,'w') as map_file:
        json.dump(map_attributes, map_file, indent=2)

def givecoord_test(robot_ekf, robot_cur, offset_rot = 0, offset_x = 0, offset_y = 0): #for testing only, will plot ground truth and aligned aruco markers
    
    gt_aruco = parse_groundtruth('TRUEMAP.txt')
    us_aruco = parse_user_map('lab_output/slam.txt')
    
    taglist, us_vec, _ = match_aruco_points_test(us_aruco, gt_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec = us_vec[:, idx]
    # gt_vec = gt_vec[:, idx]

    # theta, x = solve_umeyama2d(us_vec, gt_vec)
    # us_vec_aligned = apply_transform(theta, x, us_vec)
    
    #x = -np.array([[robot_pose[0][0]],[robot_pose[1][0]]])
    #theta = -robot_pose[2][0]
    
    theta, x = solve_trans_rot(robot_cur, robot_ekf)
    us_vec_aligned = apply_transform_custom(-theta, -x, us_vec, offset_rot, offset_x, offset_y) #rmse after aligning using robot pose
    save(us_vec_aligned)

    # diff = gt_vec - us_vec_aligned
    # rmse = compute_rmse(us_vec, gt_vec) #rmse before any alignment
    # rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)   #rmse after aligning using ground truth


    # theta_gt, x_gt = solve_umeyama2d(us_vec, gt_vec)
    # us_vec_gt_align = apply_transform(theta_gt, x_gt, us_vec)
    # diff_ori = gt_vec - us_vec_gt_align
    # rmse_ori = compute_rmse(us_vec_gt_align, gt_vec)
    # rmse_gt_align = compute_rmse(us_vec_gt_align, gt_vec)  


    # print()
    # print("The following parameters optimally transform the estimated points to the ground truth.")
    # print("Rotation Angle: {}".format(theta))
    # print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    # print()
    # print("Number of found markers: {}".format(len(taglist)))
    # print("RMSE before alignment: {}".format(rmse))
    # print("RMSE after alignment using robot pose:  {}".format(rmse_aligned))
    # print("RMSE after alignment using ground truth:  {}".format(rmse_gt_align))

    # print()
    # print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    # print('-----------------------------------------------------------------')
    # for i in range(len(taglist)):
        # print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], gt_vec[0][i], us_vec_aligned[0][i], diff[0][i], gt_vec[1][i], us_vec_aligned[1][i], diff[1][i]))
    
    ax = plt.gca()
    # ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
    ax.scatter(us_vec[0,:], us_vec[1,:], marker='x', color='C3', s=100)
    # ax.scatter(us_vec_gt_align[0,:], us_vec_gt_align[1,:], marker='x', color='C4', s=100)

    for i in range(len(taglist)):
        # ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
        ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
        ax.text(us_vec[0,i]+0.05, us_vec[1,i]+0.05, taglist[i], color='C3', size=12)
        # ax.text(us_vec_gt_align[0,i]+0.05, us_vec_gt_align[1,i]+0.05, taglist[i], color='C4', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    # plt.legend(['Real','after align','before align','gt align'])
    plt.legend(['after align','before align'])
    plt.grid()
    plt.savefig('pics/test_plot_markers.png')
    plt.close()
    
    # return rmse_aligned
    
def evaluate_after(): #for testing only, will plot ground truth and aligned aruco markers
    
    gt_aruco = parse_groundtruth('TRUEMAP.txt')
    us_aruco = parse_generated_gt('slam_aligned.txt')
    
    taglist, us_vec_aligned, gt_vec = match_aruco_points_test(us_aruco, gt_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec_aligned = us_vec_aligned[:, idx]
    gt_vec = gt_vec[:, idx]

    diff = gt_vec - us_vec_aligned
    rmse_aligned = compute_rmse(us_vec_aligned, gt_vec) 

    theta_gt, x_gt = solve_umeyama2d(us_vec_aligned, gt_vec)
    us_vec_gt_align = apply_transform(theta_gt, x_gt, us_vec_aligned)
    
    diff_ori = gt_vec - us_vec_gt_align
    rmse_gt_align = compute_rmse(us_vec_gt_align, gt_vec)  
    
    print()
    print("Number of found markers: {}".format(len(taglist)))
    print("RMSE after alignment using robot pose:  {}".format(rmse_aligned))
    print("RMSE after alignment using ground truth:  {}".format(rmse_gt_align))

def display_marker_rmse(robot_pose):
    gt_aruco = parse_groundtruth('TRUEMAP.txt')
    us_aruco = parse_user_map('lab_output/test.txt')

    taglist, us_vec, gt_vec = match_aruco_points_test(us_aruco, gt_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec = us_vec[:,idx]
    gt_vec = gt_vec[:, idx]

    # theta, x = solve_umeyama2d(us_vec, gt_vec)
    # us_vec_aligned = apply_transform(theta, x, us_vec)
    
    # x = -np.array([[robot_pose[0][0]],[robot_pose[1][0]]])
    # theta = -robot_pose[2][0]
    # us_vec_aligned = apply_transform(theta, x, us_vec)

    diff = gt_vec - us_vec_aligned
    rmse = compute_rmse(us_vec, gt_vec)
    rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)  
    print(f"RMSE error: {rmse_aligned}")

    return rmse_aligned

if __name__ == '__main__':
    evaluate_after()
    
    # import argparse

    # parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    #parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
    #parser.add_argument("estimate", type=str, help="The estimate file name.")
    #args = parser.parse_args()


    #us_aruco = parse_user_map('lab_output/slam.txt')
    # robot_pose = EKF.robot.state.squeeze().tolist()

    # theta=[robot_pose[0],robot_pose[1]]
    # x=robot_pose[2]
    # us_vec_aligned = apply_transform(theta, x, us_vec)
    

    
    # print()
    # print("The following parameters optimally transform the estimated points to the ground truth.")
    # print("Rotation Angle: {}".format(theta))
    # print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    # print()
    # print("Number of found markers: {}".format(len(taglist)))
    # # print("RMSE before alignment: {}".format(rmse))
    # # print("RMSE after alignment:  {}".format(rmse_aligned))
    
    # print()
    # print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    # print('-----------------------------------------------------------------')
    # for i in range(len(taglist)):
    #     print('%3d %9.2f %9.2f\n' % (taglist[i], us_vec_aligned[0][i], us_vec_aligned[1][i]))
    
    # ax = plt.gca()
    # #ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    # ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
    # for i in range(len(taglist)):
    #     #ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
    #     ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
    # plt.title('Arena')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    # ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    # plt.legend(['Real','Pred'])
    # plt.grid()
    # plt.show()