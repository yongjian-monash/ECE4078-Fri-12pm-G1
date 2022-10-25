# estimate the pose of a target object detected
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import ast

show_animation = True

# def parse_groundtruth(fname : str) -> dict:
#     with open(fname, 'r') as f: 
#         try:
#             gt_dict = json.load(f)                   
#         except ValueError as e:
#             with open(fname, 'r') as f:
#                 gt_dict = ast.literal_eval(f.readline()) 
        
#         aruco_dict = {}
#         for key in gt_dict:
#             if key.startswith("aruco"):
#                 aruco_num = int(key.strip('aruco')[:-2])
#                 aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
#     return aruco_dict

# def parse_user_map(fname : str) -> dict:
#     with open(fname, 'r') as f:
#         try:
#             usr_dict = json.load(f)                   
#         except ValueError as e:
#             with open(fname, 'r') as f:
#                 usr_dict = ast.literal_eval(f.readline()) 
#         aruco_dict = {}
#         for (i, tag) in enumerate(usr_dict["taglist"]):
#             aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
#     return aruco_dict

# def match_aruco_points(aruco0 : dict):
#     points0 = []
#     keys = []
#     aruco1=[1,2,3,4,5,6,7,8,9,10]
#     for key in aruco0:
#         if not key in aruco1:
#             continue
        
#         points0.append(aruco0[key])
#         keys.append(key)
#     return keys, np.hstack(points0)


# gt_aruco = parse_groundtruth('TRUEMAP.txt')
# us_aruco = parse_user_map('lab_output/test.txt')
# taglist, us_vec= match_aruco_points(us_aruco)
# print(taglist)
# print(us_vec)

# idx = np.argsort(taglist)
# taglist = np.array(taglist)[idx]
# us_vec = us_vec[:,idx]


# ax = plt.gca()
# ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
# ax.scatter(us_vec[0,:], us_vec[1,:], marker='x', color='C1', s=100)
# for i in range(len(taglist)):
#     ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
#     ax.text(us_vec[0,i]+0.05, us_vec[1,i]+0.05, taglist[i], color='C1', size=12)
# plt.title('Arena')
# plt.xlabel('X')
# plt.ylabel('Y')
# ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
# ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
# plt.legend(['Real','Pred'])
# plt.grid()
# plt.show()

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos



def main():
    print(__file__ + " start!!")
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('testing_chris.txt')
    print(fruit_list)
    print(fruit_true_pos)
    print(aruco_true_pos)
    print(aruco_true_pos[0][0])
    taglist=[1,2,3,4,5,6,7,8,9,10]

    ax = plt.gca()
    for i in range(len(aruco_true_pos)):
        p1 = ax.scatter(aruco_true_pos[i,0], aruco_true_pos[i,1], marker='x', color='C1', s=100)
        ax.text(aruco_true_pos[i,0]+0.05, aruco_true_pos[i,1]+0.05, taglist[i], color='C1', size=12)

    for i in range(len(fruit_true_pos)):
        p2 = ax.scatter(fruit_true_pos[i,0], fruit_true_pos[i,1], marker='o', color='C2', s=100)
        ax.text(fruit_true_pos[i,0]+0.05, fruit_true_pos[i,1]+0.05, fruit_list[i], color='C2', size=10)

    plt.legend([p1, p2], ["Markers","Fruits"])

    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.grid()
    plt.show()    

if __name__ == '__main__':
    main()