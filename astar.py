import numpy as np
import heapq

import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Node:

    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    ''' evaluation funcs for heapq '''
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(self.position)


class AStar:

    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid

        self.open = []
        self.closed = set()
        self.g_values = {start: 0}
        self.parents = {}
        self.history = []

    def _heuristic(self, node1, node2):
        node1_x, node1_y = node1.position
        node2_x, node2_y = node2.position
        return np.sqrt((node1_x-node2_x)**2+(node1_y-node2_y)**2)

    def _find_neighbours(self, node, grid):
        neighbours = []
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

        rows, cols = grid.shape
        x, y = node.position
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx,ny] == 0:
                neighbours.append(Node((nx, ny)))
        
        return neighbours

    def _backtrack_path(self, node):
        path = [node.position]
        while node.parent:
            path.append(node.parent.position)
            node = node.parent
        return path[::-1]

    def run(self):

        heapq.heapify(self.open)
        heapq.heappush(self.open, self.start)
        
        while self.open:
            current_node = heapq.heappop(self.open)
            current_pos = current_node.position
            self.history.append(current_pos)

            if current_node == self.goal:
                print(f'Path found with cost: {current_node.g}')
                return self._backtrack_path(current_node), self.history
            
            self.closed.add(current_node)

            for neighbour in self._find_neighbours(current_node, self.grid):
                if neighbour in self.closed:
                    continue
                
                # calculate step cost & new g
                neighbour_pos = neighbour.position
                dx = abs(neighbour_pos[0] - current_pos[0])
                dy = abs(neighbour_pos[1] - current_pos[1])
                cost = np.sqrt(2) if dx + dy == 2 else 1
                tmp_g = current_node.g + cost

                if neighbour in self.open:
                    if tmp_g < neighbour.g:
                        neighbour.g = tmp_g
                        neighbour.f = tmp_g + neighbour.h
                        neighbour.parent = current_node
                        heapq.heapify(self.open)
                else:
                    neighbour.g = tmp_g
                    neighbour.h = self._heuristic(neighbour, self.goal)
                    neighbour.f = tmp_g + neighbour.h
                    neighbour.parent = current_node
                    heapq.heappush(self.open, neighbour)

        return None, self.history

def visualise(grid, start, goal, path, history):

    fig, ax = plt.subplots(figsize=(10, 10))

    # draw grid
    rows, cols = grid.shape
    ax.imshow(grid, cmap='Greys')

    # draw start and goal
    ax.add_patch(Rectangle((start[1],start[0]), 1, 1, color='green', linewidth=0))
    ax.add_patch(Rectangle((goal[1],goal[0]), 1, 1, color='blue', linewidth=0))

    # draw path
    path_x = [pos[0] + 0.5 for pos in path]
    path_y = [pos[1] + 0.5 for pos in path]
    ax.plot(path_y, path_x, linewidth=2, alpha=0.8, label='Path', color='red')

    # draw history
    ax.scatter([pos[1] for pos in history], [pos[0] for pos in history], color='gray', alpha=0.1)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0,cols)
    ax.set_ylim(rows,0)
    plt.tight_layout()
    plt.savefig('./figs/astar.png', dpi=300)

if __name__ == '__main__':

    # open map of london and capture grid data
    with open('maps/London_2_256.map', 'r') as f:
        grid = [list(line.strip()) for line in f.readlines()]
        grid = grid[4::]

    # transform to grid for our implentation (1s and 0s)
    for i in range(256):
        for j in range(256):
            if grid[i][j] == '.' or grid[i][j] == 'G': # passable terrain
                grid[i][j] = 0
            else:
                grid[i][j] = 1

    grid = np.array(grid)
    start = (15,30) # green
    goal = (150, 250) # blue

    trial = AStar(Node(start), Node((goal)), grid)
    t1 = time.perf_counter()
    path, history = trial.run()
    t2 = time.perf_counter()

    print(f'Time taken: {t2 - t1:.5f} seconds')
    print(f'Explored nodes: {len(trial.closed)}')

    visualise(grid, start, goal, path, history)