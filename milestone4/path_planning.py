"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pd)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
Implemented maintaining similarity with the pseudocode for understanding.
Code can be significantly optimized by using a priority queue for U, etc.
Avoiding additional imports based on repository philosophy.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import ast

show_animation = False
pause_time = 0.1
p_create_random_obstacle = 0

class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost

def add_coordinates(node1: Node, node2: Node):
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node

def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y

class DStarLite:

    # Please adjust the heuristic function (h) if you change the list of
    # possible motions
    d = 2
    motions = [
        Node(d, 0, d),
        Node(0, d, d),
        Node(-d, 0, d),
        Node(0, -d, d),
        # Node(d, d, math.sqrt(2*(d**2))),
        # Node(d, -d, math.sqrt(2*(d**2))),
        # Node(-d, d, math.sqrt(2*(d**2))),
        # Node(-d, -d, math.sqrt(2*(d**2)))
    ]

    def __init__(self, ox: list, oy: list):
        # Ensure that within the algorithm implementation all node coordinates
        # are indices in the grid and extend
        # from 0 to abs(<axis>_max - <axis>_min)
        self.x_min_world = int(min(ox))
        self.y_min_world = int(min(oy))
        self.x_max = int(abs(max(ox) - self.x_min_world))
        self.y_max = int(abs(max(oy) - self.y_min_world))
        self.obstacles = [Node(x - self.x_min_world, y - self.y_min_world)
                          for x, y in zip(ox, oy)]
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = list()  # type: ignore
        self.km = 0.0
        self.kold = 0.0
        self.rhs = list()  # type: ignore
        self.g = list()  # type: ignore
        self.detected_obstacles = list()  # type: ignore
        if show_animation:
            self.detected_obstacles_for_plotting_x = list()  # type: ignore
            self.detected_obstacles_for_plotting_y = list()  # type: ignore

    def create_grid(self, val: float):
        grid = list()
        for _ in range(0, self.x_max):
            grid_row = list()
            for _ in range(0, self.y_max):
                grid_row.append(val)
            grid.append(grid_row)
        return grid

    def is_obstacle(self, node: Node):
        return any([compare_coordinates(node, obstacle)
                    for obstacle in self.obstacles]) or \
               any([compare_coordinates(node, obstacle)
                    for obstacle in self.detected_obstacles])

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            # Attempting to move from or to an obstacle
            return math.inf
        new_node = Node(node1.x-node2.x, node1.y-node2.y)
        detected_motion = list(filter(lambda motion:
                                      compare_coordinates(motion, new_node),
                                      self.motions))
        return detected_motion[0].cost

    def h(self, s: Node):
        # Cannot use the 2nd euclidean norm as this might sometimes generate
        # heuristics that overestimate the cost, making them inadmissible,
        # due to rounding errors etc (when combined with calculate_key)
        # To be admissible heuristic should
        # never overestimate the cost of a move
        # hence not using the line below
        # return math.hypot(self.start.x - s.x, self.start.y - s.y)

        # Below is the same as 1; modify if you modify the cost of each move in
        # motion
        return max(abs(self.start.x - s.x), abs(self.start.y - s.y))
        # return 1

    def calculate_key(self, s: Node):
        return (min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s)
                + self.km, min(self.g[s.x][s.y], self.rhs[s.x][s.y]))

    def is_valid(self, node: Node):
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def get_neighbours(self, u: Node):
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def pred(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def succ(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def initialize(self, start: Node, goal: Node):
        self.start.x = start.x - self.x_min_world
        self.start.y = start.y - self.y_min_world
        self.goal.x = goal.x - self.x_min_world
        self.goal.y = goal.y - self.y_min_world
        self.U = list()  # Would normally be a priority queue
        self.km = 0.0
        self.rhs = self.create_grid(math.inf)
        self.g = self.create_grid(math.inf)
        self.rhs[self.goal.x][self.goal.y] = 0
        self.U.append((self.goal, self.calculate_key(self.goal)))
        self.detected_obstacles = list()

    def update_vertex(self, u: Node):
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min([self.c(u, sprime) +
                                      self.g[sprime.x][sprime.y]
                                      for sprime in self.succ(u)])
        if any([compare_coordinates(u, node) for node, key in self.U]):
            self.U = [(node, key) for node, key in self.U
                      if not compare_coordinates(node, u)]
            self.U.sort(key=lambda x: x[1])
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key_pair1: tuple[float, float],
                     key_pair2: tuple[float, float]):
        return key_pair1[0] < key_pair2[0] or \
               (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        self.U.sort(key=lambda x: x[1])
        while (len(self.U) > 0 and
               self.compare_keys(self.U[0][1],
                                 self.calculate_key(self.start))) or \
                self.rhs[self.start.x][self.start.y] != \
                self.g[self.start.x][self.start.y]:
            self.kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            if self.compare_keys(self.kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x][u.y] > self.rhs[u.x][u.y]:
                self.g[u.x][u.y] = self.rhs[u.x][u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x][u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)
            self.U.sort(key=lambda x: x[1])

    def detect_changes(self):
        changed_vertices = list()
        if len(self.spoofed_obstacles) > 0:
            for spoofed_obstacle in self.spoofed_obstacles[0]:
                if compare_coordinates(spoofed_obstacle, self.start) or \
                   compare_coordinates(spoofed_obstacle, self.goal):
                    continue
                changed_vertices.append(spoofed_obstacle)
                self.detected_obstacles.append(spoofed_obstacle)
                if show_animation:
                    self.detected_obstacles_for_plotting_x.append(
                        spoofed_obstacle.x + self.x_min_world)
                    self.detected_obstacles_for_plotting_y.append(
                        spoofed_obstacle.y + self.y_min_world)
                    plt.plot(self.detected_obstacles_for_plotting_x,
                             self.detected_obstacles_for_plotting_y, ".k")
                    plt.pause(pause_time)
            self.spoofed_obstacles.pop(0)

        # Allows random generation of obstacles
        random.seed()
        if random.random() > 1 - p_create_random_obstacle:
            x = random.randint(0, self.x_max)
            y = random.randint(0, self.y_max)
            new_obs = Node(x, y)
            if compare_coordinates(new_obs, self.start) or \
               compare_coordinates(new_obs, self.goal):
                return changed_vertices
            changed_vertices.append(Node(x, y))
            self.detected_obstacles.append(Node(x, y))
            if show_animation:
                self.detected_obstacles_for_plotting_x.append(x +
                                                              self.x_min_world)
                self.detected_obstacles_for_plotting_y.append(y +
                                                              self.y_min_world)
                plt.plot(self.detected_obstacles_for_plotting_x,
                         self.detected_obstacles_for_plotting_y, ".k")
                plt.pause(pause_time)
        return changed_vertices

    def compute_current_path(self):
        path = list()
        current_point = Node(self.start.x, self.start.y)
        while not compare_coordinates(current_point, self.goal):
            path.append(current_point)
            current_point = min(self.succ(current_point),
                                key=lambda sprime:
                                self.c(current_point, sprime) +
                                self.g[sprime.x][sprime.y])
        path.append(self.goal)
        return path

    def compare_paths(self, path1: list, path2: list):
        if len(path1) != len(path2):
            return False
        for node1, node2 in zip(path1, path2):
            if not compare_coordinates(node1, node2):
                return False
        return True

    def display_path(self, path: list, colour: str, alpha: float = 1.0):
        px = [(node.x + self.x_min_world) for node in path]
        py = [(node.y + self.y_min_world) for node in path]
        drawing = plt.plot(px, py, colour, alpha=alpha)
        plt.pause(pause_time)
        return drawing

    def main(self, start: Node, goal: Node,
             spoofed_ox: list, spoofed_oy: list):
        self.spoofed_obstacles = [[Node(x - self.x_min_world,
                                        y - self.y_min_world)
                                   for x, y in zip(rowx, rowy)]
                                  for rowx, rowy in zip(spoofed_ox, spoofed_oy)
                                  ]
        pathx = []
        pathy = []
        self.initialize(start, goal)
        last = self.start
        self.compute_shortest_path()
        pathx.append(self.start.x + self.x_min_world)
        pathy.append(self.start.y + self.y_min_world)

        if show_animation:
            current_path = self.compute_current_path()
            previous_path = current_path.copy()
            previous_path_image = self.display_path(previous_path, ".c",
                                                    alpha=0.3)
            current_path_image = self.display_path(current_path, ".c")

        while not compare_coordinates(self.goal, self.start):
            if self.g[self.start.x][self.start.y] == math.inf:
                print("No path possible")
                return False, pathx, pathy
            self.start = min(self.succ(self.start),
                             key=lambda sprime:
                             self.c(self.start, sprime) +
                             self.g[sprime.x][sprime.y])
            pathx.append(self.start.x + self.x_min_world)
            pathy.append(self.start.y + self.y_min_world)
            if show_animation:
                current_path.pop(0)
                plt.plot(pathx, pathy, "-r")
                plt.pause(pause_time)
            changed_vertices = self.detect_changes()
            if len(changed_vertices) != 0:
                print("New obstacle detected")
                self.km += self.h(last)
                last = self.start
                for u in changed_vertices:
                    if compare_coordinates(u, self.start):
                        continue
                    self.rhs[u.x][u.y] = math.inf
                    self.g[u.x][u.y] = math.inf
                    self.update_vertex(u)
                self.compute_shortest_path()

                if show_animation:
                    new_path = self.compute_current_path()
                    if not self.compare_paths(current_path, new_path):
                        current_path_image[0].remove()
                        previous_path_image[0].remove()
                        previous_path = current_path.copy()
                        current_path = new_path.copy()
                        previous_path_image = self.display_path(previous_path,
                                                                ".c",
                                                                alpha=0.3)
                        current_path_image = self.display_path(current_path,
                                                               ".c")
                        plt.pause(pause_time)
        print("Path found")
        return True, pathx, pathy


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))
        
def generate_obstacles(fruit_true_pos, aruco_true_pos):
    ox, oy = [], []
    
    # generate obstacles for map boundaries
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
        
    # generate obstacles for aruco markers
    for i in aruco_true_pos:
        ox.append(int(i[0] * 10))
        oy.append(int(i[1] * 10))
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            ox.append(j)
            oy.append(int(i[1] * 10) - 1)
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            ox.append(j)
            oy.append(int(i[1] * 10) + 1)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            ox.append(int(i[0] * 10) - 1)
            oy.append(j)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            ox.append(int(i[0] * 10) + 1)
            oy.append(j)
            
    # generate obstacles for fruits
    for i in fruit_true_pos:
        ox.append(int(i[0] * 10))
        oy.append(int(i[1] * 10))
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            ox.append(j)
            oy.append(int(i[1] * 10) - 1)
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            ox.append(j)
            oy.append(int(i[1] * 10) + 1)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            ox.append(int(i[0] * 10) - 1)
            oy.append(j)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            ox.append(int(i[0] * 10) + 1)
            oy.append(j)
    
    return ox, oy

def generate_points_L2(fruit_goals, aruco_true_pos):
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
            x = round_nearest(fruit_goals[i][0] + x_list[k], 0.2)
            y = round_nearest(fruit_goals[i][1] + y_list[k], 0.2)
            
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
    fx = (fruit_goals * 10).astype(int)[:, 0]
    fy = (fruit_goals * 10).astype(int)[:, 1]
    
    return sx, sy, gx, gy, fx, fy, face_angle

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
        for i in range(3):
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

def generate_spoofed_obs(spoofed_obs):
    # spoofed_obs should be a list of lists
    # inner list should be [x, y]
    spoofed_ox = []
    spoofed_oy = []

    for i in spoofed_obs:
        spoofed_ox.append(int(i[0] * 10))
        spoofed_oy.append(int(i[1] * 10))
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            spoofed_ox.append(j)
            spoofed_oy.append(int(i[1] * 10) - 1)
        for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 2):
            spoofed_ox.append(j)
            spoofed_oy.append(int(i[1] * 10) + 1)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            spoofed_ox.append(int(i[0] * 10) - 1)
            spoofed_oy.append(j)
        for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 2):
            spoofed_ox.append(int(i[0] * 10) + 1)
            spoofed_oy.append(j)

    return spoofed_ox, spoofed_oy

class GenerateCoord:
    def __init__(self, fname):
        self.fname = fname
    
    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def read_true_map(self):
        """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

        @param fname: filename of the map
        @return:
            1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
            2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        self.fruit_list = []
        self.fruit_true_pos = []
        self.aruco_true_pos = np.empty([10, 2])

        with open(self.fname, 'r') as f:
            try:
                gt_dict = json.load(f)
            except ValueError as e:
                with open(self.fname, 'r') as f:
                    gt_dict = ast.literal_eval(f.readline())

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        self.aruco_true_pos[9][0] = x
                        self.aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5])
                        self.aruco_true_pos[marker_id-1][0] = x
                        self.aruco_true_pos[marker_id-1][1] = y
                else:
                    self.fruit_list.append(key[:-2])
                    if len(self.fruit_true_pos) == 0:
                        self.fruit_true_pos = np.array([[x, y]])
                    else:
                        self.fruit_true_pos = np.append(self.fruit_true_pos, [[x, y]], axis=0)

            return self.fruit_list, self.fruit_true_pos, self.aruco_true_pos

    def generate_points(self, spoofed_obs):
        self.fruit_list, self.fruit_true_pos, self.aruco_true_pos = self.read_true_map()
        sx = np.array([0])
        sy = np.array([0])
        new_goal = np.zeros(self.fruit_true_pos.shape)
        face_angle = np.zeros(len(self.fruit_true_pos))

        # start and goal position
        for i in range(len(self.fruit_true_pos)):
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
                x = self.round_nearest(self.fruit_true_pos[i][0] + x_list[k], 0.2)
                y = self.round_nearest(self.fruit_true_pos[i][1] + y_list[k], 0.2)
                
                if not (np.array([x, y]) == self.aruco_true_pos).all(1).any() and (not ((np.array([x, y]) == spoofed_obs).all(1).any()) if len(spoofed_obs) != 0 else True):
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
        fx = (self.fruit_true_pos * 10).astype(int)[:, 0]
        fy = (self.fruit_true_pos * 10).astype(int)[:, 1]

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
        for i in self.aruco_true_pos:
            ox.append(int(i[0] * 10))
            oy.append(int(i[1] * 10))
            for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
                ox.append(j)
                oy.append(int(i[1] * 10) - 1)
            for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
                ox.append(j)
                oy.append(int(i[1] * 10) + 1)
            for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
                ox.append(int(i[0] * 10) - 1)
                oy.append(j)
            for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
                ox.append(int(i[0] * 10) + 1)
                oy.append(j)
        for i in self.fruit_true_pos:
            ox.append(int(i[0] * 10))
            oy.append(int(i[1] * 10))
            for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
                ox.append(j)
                oy.append(int(i[1] * 10) - 1)
            for j in range(int(i[0] * 10) - 1, int(i[0] * 10) + 1 + 1):
                ox.append(j)
                oy.append(int(i[1] * 10) + 1)
            for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
                ox.append(int(i[0] * 10) - 1)
                oy.append(j)
            for j in range(int(i[1] * 10) - 1, int(i[1] * 10) + 1 + 1):
                ox.append(int(i[0] * 10) + 1)
                oy.append(j)

        return sx, sy, gx, gy, fx, fy, ox, oy, face_angle

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
        
def animation():
    gen_cor = GenerateCoord('M4_true_map.txt')
    fruit_list, _, _ = gen_cor.read_true_map()
    obs_fruit_list = []
    spoofed_obs = []
    spoofed_ox = [[],[],[],[]]
    spoofed_oy = [[],[],[],[]]
    prev_len = len(spoofed_obs)
    
    sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(spoofed_obs)
    if show_animation:
        plt.figure(figsize=(4.8,4.8))
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(fx, fy, ".r")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Fruits', 'Path taken',
                        'Current computed path', 'Previous computed path',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['.', 'r', 1],
                                                 ['-', 'r', 1],
                                                 ['.', 'c', 1],
                                                 ['.', 'c', 0.3],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:", fontsize="xx-small")
        plt.plot()
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.xticks(np.arange(-20, 21, 4))
        plt.yticks(np.arange(-20, 21, 4))
        # plt.show()
        plt.pause(pause_time)
        
    dstarlite = DStarLite(ox, oy)

    waypoints_x = []
    waypoints_y = []
    for i in range(len(sx)):
        _, pathx, pathy = dstarlite.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)
        pathx.pop(0)
        pathy.pop(0)
        waypoints_x.extend(pathx)
        waypoints_y.extend(pathy)
    waypoints_x = [x/10.0 for x in waypoints_x]
    waypoints_y = [y/10.0 for y in waypoints_y]
    waypoints_list = [[x,y] for x, y in zip(waypoints_x, waypoints_y)]
        
    plt.show()

def ori():

    # start and goal position
    sx = 10  # [m]
    sy = 10  # [m]
    gx = 50  # [m]
    gy = 50  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Path taken',
                        'Current computed path', 'Previous computed path',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['-', 'r', 1],
                                                 ['.', 'c', 1],
                                                 ['.', 'c', 0.3],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:",
                   fontsize="xx-small")
        plt.plot()
        plt.pause(pause_time)

    # Obstacles discovered at time = row
    # time = 1, obstacles discovered at (0, 2), (9, 2), (4, 0)
    # time = 2, obstacles discovered at (0, 1), (7, 7)
    # ...
    # when the spoofed obstacles are:
    # spoofed_ox = [[0, 9, 4], [0, 7], [], [], [], [], [], [5]]
    # spoofed_oy = [[2, 2, 0], [1, 7], [], [], [], [], [], [4]]

    # Reroute
    # spoofed_ox = [[], [], [], [], [], [], [], [40 for _ in range(10, 21)]]
    # spoofed_oy = [[], [], [], [], [], [], [], [i for i in range(10, 21)]]

    # Obstacles that demostrate large rerouting
    spoofed_ox = [[], [], [],
                  [i for i in range(0, 21)] + [0 for _ in range(0, 20)]]
    spoofed_oy = [[], [], [],
                  [20 for _ in range(0, 21)] + [i for i in range(0, 20)]]

    dstarlite = DStarLite(ox, oy)
    dstarlite.main(Node(x=sx, y=sy), Node(x=gx, y=gy),
                   spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)
                   
def new_main():
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
    search_list = read_search_list()
    fruit_goals = print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)
    
    ox, oy = generate_obstacles(fruit_true_pos, aruco_true_pos)
    dstarlite = DStarLite(ox, oy)
    
    sx, sy, gx, gy, fx, fy, face_angle = generate_points_L2(fruit_goals, aruco_true_pos)
    
    if show_animation:
        plt.figure(figsize=(4.8,4.8))
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(fx, fy, ".r")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Fruits', 'Path taken',
                        'Current computed path', 'Previous computed path',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['.', 'r', 1],
                                                 ['-', 'r', 1],
                                                 ['.', 'c', 1],
                                                 ['.', 'c', 0.3],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:", fontsize="xx-small")
        plt.plot()
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.xticks(np.arange(-20, 21, 4))
        plt.yticks(np.arange(-20, 21, 4))
        # plt.show()
        plt.pause(pause_time)
        
        
    
    waypoints_list = []
    for i in range(len(sx)):
        _, pathx, pathy = dstarlite.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=[], spoofed_oy=[])
        pathx.pop(0)
        pathy.pop(0)
        temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
        waypoints_list.append(temp)
    print(waypoints_list)
        
    plt.show()

if __name__ == "__main__":
    show_animation = True
    pause_time = 0.1
    p_create_random_obstacle = 0
    new_main()