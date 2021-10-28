import sys
sys.path.insert(0, '')
sys.path.extend(['../'])
from feeders.specs import get_neighbours, openpose_joints, holistic_joints

import numpy as np

from graph import tools

def get_neighbour(num_node):
    self_link = [(i, i) for i in range(num_node)]
    inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
    inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward
    return neighbor

num_node = 25
neighbor = get_neighbour(num_node)


class AdjMatrixGraph:
    def __init__(self, num_node=None, *args, body_type="SMPLH", **kwargs):
        if num_node is not None:
            self.edges = get_neighbours(joints=holistic_joints, body_type=body_type, num_node=num_node)#joints=None, reduced=num_node <= 30)
        else:
            self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
