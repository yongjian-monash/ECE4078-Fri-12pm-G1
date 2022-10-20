"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import ast

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        d=1
        motion = [
                  [d, 0, d],
                  [0, d, d],
                  [-d, 0, d],
                  [0, -d, d],
                  [-d, -d, math.sqrt(2*(d**2))],
                  [-d, d, math.sqrt(2*(d**2))],
                  [d, -d, math.sqrt(2*(d**2))],
                  [d, d, math.sqrt(2*(d**2))]]

        return motion

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

def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    fruit_goals = []
    for fruit in search_list:
        for i in range(5):
            if fruit == fruit_list[i]:
                x = np.round(fruit_true_pos[i][0], 1)
                y = np.round(fruit_true_pos[i][1], 1)
                print('{}) {} at [{}, {}]'.format(n_fruit, fruit, x, y))
                if len(fruit_goals) == 0:
                    fruit_goals = np.array([[x, y]])
                else:
                    fruit_goals = np.append(fruit_goals, [[x, y]], axis=0)
        n_fruit += 1
    
    return fruit_goals


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))
    
def astar_generate(fruit_obs,aruco_obs):
    # start and goal position
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    # gx = 50.0  # [m]
    # gy = 50.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-16, 16+1):
        ox.append(i)
        oy.append(-16)
    for i in range(-16, 16+1):
        ox.append(16)
        oy.append(i)
    for i in range(-16, 16+1):
        ox.append(i)
        oy.append(16)
    for i in range(-16, 16+1):
        ox.append(-16)
        oy.append(i)

    for coord in aruco_obs:
        ox.append(coord[0]*10)
        oy.append(coord[1]*10)
        for j in range(int(coord[0] * 10) - 1, int(coord[0] * 10) + 2):
            ox.append(j)
            oy.append(int(coord[1] * 10) - 1)
        for j in range(int(coord[0] * 10) - 1, int(coord[0] * 10) + 2):
            ox.append(j)
            oy.append(int(coord[1] * 10) + 1)
        for j in range(int(coord[1] * 10) - 1, int(coord[1] * 10) + 2):
            ox.append(int(coord[0] * 10) - 1)
            oy.append(j)
        for j in range(int(coord[1] * 10) - 1, int(coord[1] * 10) + 2):
            ox.append(int(coord[0] * 10) + 1)
            oy.append(j)         

    for coord in fruit_obs:
        ox.append(coord[0]*10)
        oy.append(coord[1]*10)
        for j in range(int(coord[0] * 10) - 1, int(coord[0] * 10) + 2):
            ox.append(j)
            oy.append(int(coord[1] * 10) - 1)
        for j in range(int(coord[0] * 10) - 1, int(coord[0] * 10) + 2):
            ox.append(j)
            oy.append(int(coord[1] * 10) + 1)
        for j in range(int(coord[1] * 10) - 1, int(coord[1] * 10) + 2):
            ox.append(int(coord[0] * 10) - 1)
            oy.append(j)
        for j in range(int(coord[1] * 10) - 1, int(coord[1] * 10) + 2):
            ox.append(int(coord[0] * 10) + 1)
            oy.append(j)    
    


    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

    return ox, oy, a_star

def generate_points(fruit_goals, aruco_true_pos, fruit_true_pos):
    sx = np.array([0])
    sy = np.array([0])
    new_goal = np.zeros(fruit_goals.shape)
    face_angle = np.zeros(len(fruit_goals))

    # start and goal position
    for i in range(len(fruit_goals)):
        possible_goal = []
        possible_angle = []
        
        # start from north, in 8 compass position away from goal
        d = 0.2 # distance from goal
        x_list = [-d, -d, 0, d, d, d, 0, -d]
        y_list = [0, -d, -d, -d, 0, d, d, d]
        angle_list = [0, 0.25*np.pi, 0.50*np.pi, 0.75*np.pi, 1.00*np.pi, -0.75*np.pi, -0.50*np.pi, -0.25*np.pi]
        
        # # start from north, in 4 compass position away from goal
        # d = 0.2 # distance from goal
        # x_list = [-d, 0, d, 0]
        # y_list = [0, -d, 0, d]
        # angle_list = [0, 0.50*np.pi, 1.00*np.pi, -0.50*np.pi]
        
        for k in range(len(x_list)):
            x = round_nearest(fruit_goals[i][0] + x_list[k], 0.1)
            y = round_nearest(fruit_goals[i][1] + y_list[k], 0.1)

            if not (np.array([x, y]) == aruco_true_pos).all(1).any():
                possible_goal.append(np.array([x, y]))
                possible_angle.append(angle_list[k])

        min_val = 10000
        for j in range(len(possible_goal)):
            dis = np.hypot(abs(sx[i] - possible_goal[j][0]), abs(sy[i] - possible_goal[j][1]))
            if dis < min_val:
                min_val = dis
                new_goal[i] = possible_goal[j]
                face_angle[i] = possible_angle[j]
        sx = np.append(sx, new_goal[i][0])
        sy = np.append(sy, new_goal[i][1])
    sx = (sx * 10).astype(int)
    sy = (sy * 10).astype(int)
    sx = np.delete(sx, -1)
    sy = np.delete(sy, -1)

    gx = (new_goal * 10).astype(int)[:, 0]  # [m]
    gy = (new_goal * 10).astype(int)[:, 1]  # [m]

    fx = (fruit_true_pos * 10).astype(int)[:, 0]
    fy = (fruit_true_pos * 10).astype(int)[:, 1]

    
    return sx, sy, gx, gy, fx, fy

def generate_spoofed_obs(self, spoofed_obs):
    self.spoofed_obs_x = []
    self.spoofed_obs_y = []

    for i in spoofed_obs:
        self.spoofed_obs_x.append(int(i[0] * 10))
        self.spoofed_obs_y.append(int(i[1] * 10))
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
            self.spoofed_obs_x.append(j)
            self.spoofed_obs_y.append(int(i[1] * 10) - 1)
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
            self.spoofed_obs_x.append(j)
            self.spoofed_obs_y.append(int(i[1] * 10) + 1)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
            self.spoofed_obs_x.append(int(i[0] * 10) - 1)
            self.spoofed_obs_y.append(j)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
            self.spoofed_obs_x.append(int(i[0] * 10) + 1)
            self.spoofed_obs_y.append(j)
    spoofed_ox = [[], [], [],
                    self.spoofed_obs_x]
    spoofed_oy = [[], [], [],
                    self.spoofed_obs_y]

    return spoofed_ox, spoofed_oy

def main():
    print(__file__ + " start!!")

    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('aruco_fruit_final.txt')
    search_list = read_search_list()
    
    fruit_goals = print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)

   
    ox, oy, a_star = astar_generate(fruit_true_pos,aruco_true_pos)

    sx, sy, gx, gy, fx, fy= generate_points(fruit_goals, aruco_true_pos,fruit_true_pos)

    waypoints_list = []
    for i in range(len(sx)):
        rx, ry = a_star.planning(sx[i], sy[i], gx[i], gy[i])
        plt.plot(rx,ry,"-r")
        rx.pop(0)
        ry.pop(0)
        temp = [[x/10.0,y/10.0] for x, y in zip(rx, ry)]
        waypoints_list.append(temp)

    print(waypoints_list)

    if show_animation:

        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(fx, fy, ".r")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Fruits', 'Path taken',
                        'Current computed path','Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                    for symbol, colour, alpha in [['o', 'g', 1],
                                                    ['x', 'b', 1],
                                                    ['.', 'r', 1],
                                                    ['-', 'r', 1],
                                                    ['.', 'c', 1],
                                                    ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:", fontsize="xx-small")
        plt.plot()
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.xticks(np.arange(-20, 21, 4))
        plt.yticks(np.arange(-20, 21, 4))
    plt.show()


if __name__ == '__main__':
    main()
