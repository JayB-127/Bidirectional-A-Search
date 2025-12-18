import numpy as np
from heapdict import heapdict
import math

from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BiAStar:
    """
    Bidirectional A* pathfinding algorithm on a 2D grid.

    Attributes:
        grid (np.ndarray): 2D array representing the map, where 0s represent free space and 1 represent obstacles.
        start (tuple[int, int]): Starting node as (row, col).
        goal (tuple[int, int]): Goal node as (row, col).
        open_fwd (heapdict(tuple[int, int], float)): Priority queue of open nodes for forward search.
        open_bwd (heapdict(tuple[int, int], float)): Priority queue of open nodes for backward search.
        closed_fwd (set(tuple[int, int])): Set of closed nodes for forward search.
        closed_bwd (set(tuple[int, int])): Set of closed nodes for backward search.
        g_fwd (dict(tuple[int, int], float)): Dictionary of g-scores for nodes in forward search.
        g_bwd (dict(tuple[int, int], float)): Dictionary of g-scores for nodes in backward search.
        parents_fwd (dict(tuple[int, int], tuple[int, int])): Dictionary mapping nodes to their parents in forward search.
        parents_bwd (dict(tuple[int, int], tuple[int, int])): Dictionary mapping nodes to their parents in backward search.
        meeting_node (tuple[int, int]): Node at which the two search directions meet.
        best_path_cost (float): Cost of the current best path found.

    """
    def __init__(self, grid, start, goal):
        """
        Initialises the bidirectional A* algorithm.

        Args:
            grid (np.ndarray): 2D array representing the map, where 0s represent free space and 1 represent obstacles.
            start (tuple[int, int]): Starting node as (row, col).
            goal (tuple[int, int]): Goal node as (row, col).
        """

        self.grid = grid
        self.start = start
        self.goal = goal

        # heapdict allows for a priority queue that uses key-decrease operations (using f score as priority)
        self.open_fwd = heapdict()
        self.open_bwd = heapdict()
        self.closed_fwd = set()
        self.closed_bwd = set()

        self.g_fwd = {start: 0}
        self.g_bwd = {goal: 0}
        self.parents_fwd = {}
        self.parents_bwd = {}

        self.meeting_node = None
        self.best_path_cost = math.inf

    def _heuristic(self, node1, node2):
        """
        Compute the Euclidean heuristic between two nodes.

        Args:
            node1 (tuple[int, int]): First node as (row, col).
            node2 (tuple[int, int]): Second node as (row, col).

        Returns:
            float: Euclidean distance between node1 and node2.
        """
        return math.hypot(node1[0]-node2[0], node1[1]-node2[1])

    def _find_neighbours(self, node):
        """
        Find all valid 8-connected neighbouring cells around a node.

        Args:
            node (tuple[int, int]): Node as (row, col).
        
        Returns:
            list[tuple[int, int]]: List of traversable neighbouring nodes.
        """
        neighbours = []
        # 8-connected directions
        directions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1)
        ]

        rows, cols = self.grid.shape
        for dx, dy in directions:
            nx = node[0] + dx
            ny = node[1] + dy
            # check neighbour is within grid bounds and it traversable
            if 0 <= nx < rows and 0 <= ny < cols and self.grid[nx,ny] == 0:
                neighbours.append((nx, ny))
        
        return neighbours
    
    def _backtrack_paths(self):
        """
        Reconstruct the forward and backward paths using parents from the meeting node.

        Returns:
            tuple[list[tuple[int, int]], list[tuple[int, int]]]: Tuple of forward (start to meeting node) and backward (meeting node to goal) paths.
        """
        
        # fwd path
        fwd_path = []
        node = self.meeting_node
        # backtrack using parents
        while node != self.start:
            fwd_path.append(node)
            node = self.parents_fwd[node]
        fwd_path.append(self.start)
        fwd_path.reverse() # currently meeting -> start so needs to be reversed

        # bwd path
        bwd_path = []
        node = self.meeting_node
        # backtrack using parents
        while node != self.goal:
            bwd_path.append(node)
            node = self.parents_bwd[node]
        bwd_path.append(self.goal)

        return (fwd_path, bwd_path)

    def run(self):
        """
        Execute the bidirectional A* search, expanding the forward and backward searches based on the size of their open sets. The search terminates when the cost of the best-known path cannot be improved.

        Returns:
            tuple[list[tuple[int, int]], list[tuple[int, int]]] or None: The whole path if found otherwise None.
        """

        # initialise priority queues with start and goal nodes
        self.open_fwd[self.start] = self._heuristic(self.start, self.goal)
        self.open_bwd[self.goal] = self._heuristic(self.goal, self.start)

        while self.open_fwd and self.open_bwd:

            # expand based on size of open lists
            if len(self.open_fwd) < len(self.open_bwd):

                # --- fwd search ---
                cur_node, cur_f = self.open_fwd.popitem()
                if cur_node in self.closed_fwd:
                    continue
                self.closed_fwd.add(cur_node)

                # check for meeting in middle, update path cost and meeting node if improved
                if cur_node in self.closed_bwd:
                    path_cost = self.g_fwd[cur_node] + self.g_bwd[cur_node]
                    if path_cost < self.best_path_cost:
                        self.best_path_cost = path_cost
                        self.meeting_node = cur_node

                # expand neighbours
                for neighbour in self._find_neighbours(cur_node):
                    # calculate step cost & new g
                    dx = abs(neighbour[0] - cur_node[0])
                    dy = abs(neighbour[1] - cur_node[1])
                    cost = np.sqrt(2) if dx + dy == 2 else 1
                    tmp_g = self.g_fwd[cur_node] + cost

                    if tmp_g < self.g_fwd.get(neighbour, math.inf):
                        # update values if node already in open set, otherwise add it
                        self.g_fwd[neighbour] = tmp_g
                        self.parents_fwd[neighbour] = cur_node
                        new_f = tmp_g + self._heuristic(neighbour, self.goal)
                        self.open_fwd[neighbour] = new_f
                
            else:
                
                # --- bwd search ---
                cur_node, cur_f = self.open_bwd.popitem()
                if cur_node in self.closed_bwd:
                    continue
                self.closed_bwd.add(cur_node)

                # check for meeting in middle, update path cost and meeting node if improved
                if cur_node in self.closed_fwd:
                    path_cost = self.g_bwd[cur_node] + self.g_fwd[cur_node]
                    if path_cost < self.best_path_cost:
                        self.best_path_cost = path_cost
                        self.meeting_node = cur_node

                # expand neighbours
                for neighbour in self._find_neighbours(cur_node):
                    # calculate step cost & new g
                    dx = abs(neighbour[0] - cur_node[0])
                    dy = abs(neighbour[1] - cur_node[1])
                    cost = np.sqrt(2) if dx + dy == 2 else 1
                    tmp_g = self.g_bwd[cur_node] + cost

                    if tmp_g < self.g_bwd.get(neighbour, math.inf):
                        # update values if node already in open set, otherwise add it
                        self.g_bwd[neighbour] = tmp_g
                        self.parents_bwd[neighbour] = cur_node
                        new_f = tmp_g + self._heuristic(neighbour, self.start)
                        self.open_bwd[neighbour] = new_f

            # termination condition: when best path cost <= best f value of remaining open nodes
            # i.e. guaranteed that both fwd and bwd search cannot find a cheaper solution
            _, min_fwd_f = self.open_fwd.peekitem()
            _, min_bwd_f = self.open_bwd.peekitem()
            if self.best_path_cost <= max(min_fwd_f, min_bwd_f) and self.meeting_node:
                return self._backtrack_paths()

        return None

def visualise(name, grid, start, goal, paths, histories, meeting):
    """
    Visualise the results of the algorithm on a 2D grid.
    
    Args:
        name (String): name of the figure to be saved.
        grid (np.ndarray): 2D map (0 = free, 1 = obstacle).
        start (tuple[int, int]): Starting node.
        goal (tuple[int, int]): Goal node.
        paths (tuple[list[tuple[int, int]], list[tuple[int, int]]]): Forward and backward paths.
        histories (tuple[set, set]): Explored nodes for forward and backward searches.
        meeting (tuple[int, int]): Node where the searches meet.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # draw grid
    rows, cols = grid.shape
    ax.imshow(grid, cmap='Greys')

    # draw start and goal
    ax.add_patch(Rectangle((start[1],start[0]), 1, 1, color='green', linewidth=0))
    ax.add_patch(Rectangle((goal[1],goal[0]), 1, 1, color='blue', linewidth=0))

    # draw paths
    fwd_x = [pos[0] + 0.5 for pos in paths[0]]
    fwd_y = [pos[1] + 0.5 for pos in paths[0]]
    ax.plot(fwd_y, fwd_x, linewidth=2, alpha=0.8, label='fwd path', color='darkgreen')
    bwd_x = [pos[0] + 0.5 for pos in paths[1]]
    bwd_y = [pos[1] + 0.5 for pos in paths[1]]
    ax.plot(bwd_y, bwd_x, linewidth=2, alpha=0.8, label='bwd path', color='darkblue')

    # draw histories
    ax.scatter([pos[1] for pos in histories[0]], [pos[0] for pos in histories[0]], color='lightgreen', alpha=0.1)
    ax.scatter([pos[1] for pos in histories[1]], [pos[0] for pos in histories[1]], color='lightblue', alpha=0.1)
    
    # draw meeting point
    ax.add_patch(Rectangle((meeting[1], meeting[0]), 1, 1, color='red', linewidth=0))
    
    # set properties and save figure
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0,cols)
    ax.set_ylim(rows,0)
    plt.tight_layout()
    plt.savefig(f'./figs/{name}.png', dpi=300)

def run_trial(start, goal, grid, name='BiAStar'):
    """
    Run trials, outputting and visualising results
    
    start (tuple[int, int]): Start node as (row, col).
    goal (tuple[int, int]): Goal node as (row, col).
    grid (np.ndarray): 2D array representing the map, where 0s represent free space and 1 represent obstacles.
    """
    # time computation time of algorithm
    trial = BiAStar(grid, start, goal)
    t1 = perf_counter()
    paths = trial.run()
    t2 = perf_counter()

    # print results
    print(f'--- {name} ---')
    print(f'Path found with cost: {trial.best_path_cost}')
    print(f'Time taken: {t2 - t1:.5f} seconds')
    print(f'Explored nodes: fwd search: {len(trial.closed_fwd)}, bwd search: {len(trial.closed_bwd)}, set: {len(trial.closed_fwd.union(trial.closed_bwd))}')

    # visualise results
    visualise(name, grid, start, goal, paths, (trial.closed_fwd, trial.closed_bwd), trial.meeting_node)

if __name__ == '__main__':

    """
    To run the algorithm:
        1.) create a map grid (0 = traversable, 1 = obstacle)
        2.) define the start and goal nodes
        3.) parse into the run_trial() and await results
        4.) statistics will be outputted to the console, and graphs will be saved using the trial name
    *Three test cases are already defined below, using different start and goal nodes on a 256 x 256 grid map of a section of London.*
    """

    # open map and capture grid data
    with open('maps/London_2_256.map', 'r') as f:
        grid = [list(line.strip()) for line in f.readlines()]
        grid = grid[4::]

    # transform to grid for our implentation
    for i in range(256):
        for j in range(256):
            if grid[i][j] == '.' or grid[i][j] == 'G': # passable terrain
                grid[i][j] = 0
            else:
                grid[i][j] = 1
    grid = np.array(grid)

    # --- TEST CASE 1 ---
    start = (15,30)
    goal = (150, 250)
    run_trial(start, goal, grid, name='Test Case 1')

    # --- TEST CASE 2 ---
    start = (245, 10)
    goal = (15, 230)
    run_trial(start, goal, grid, name='Test Case 2')

    # --- TEST CASE 3 ---
    start = (247, 215)
    goal = (7, 150)
    run_trial(start, goal, grid, name='Test Case 3')

