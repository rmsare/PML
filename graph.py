import networkx as nx
import numpy as np
from numpy.linalg import inv
from copy import deepcopy

class pml_node(object):

    def __init__(self):

        self.center = np.array([0.0, 0.0, 0.0]);
        self.transform_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 1.0]])
        self.dimensions = np.array([0.0, 0.0, 0.0])
        self.residual = None
        self.node_number = None

class pml_graph(object):

    def __init__(self, center):

        self.__G = nx.DiGraph()
        self.__current_node_number = 0
        self.__center = center

    def add_node(self, center, transform_matrix, dimensions, residual, from_node = None):

        new_node = pml_node()
        new_node.center = deepcopy(center)
        new_node.transform_matrix = deepcopy(transform_matrix)
        new_node.dimensions = deepcopy(dimensions)
        new_node.node_number = deepcopy(self.__current_node_number)
        new_node.residual = deepcopy(residual)

        self.__G.add_node(self.__current_node_number, node_info = new_node)
        self.__current_node_number += 1

        if from_node is not None:
            self.__G.add_edge(from_node.node_number, new_node.node_number)

        return new_node

    def graph(self):
        return self.__G

    def calculate_displacements(self):

        def get_external_nodes():
            return [x[1]['node_info'] for x in self.__G.nodes(data=True) if self.__G.out_degree(x[0]) == 0]

        def get_ancestors(node):

            if self.__G.in_degree(node.node_number) != 0:
                return [node] + [ancestor for edge_node in list(self.__G.in_edges(node.node_number)) for ancestor in
                                 get_ancestors(self.__G.nodes(data=True)[edge_node[0]]['node_info'])]

            return [node]

        def calculate_displacement(node, max_residual = 1.0):

            ancestors = get_ancestors(node)

            location = deepcopy(np.array([node.center]))

            for ancestor in ancestors:
                location -= ancestor.center
                location = np.concatenate((location,np.ones((1,1))), axis=1)
                location = np.matmul(location, inv(ancestor.transform_matrix).T)[:,0:3]
                location += ancestor.center

            displacement = np.array([node.center]) - location
            return displacement[0,0], displacement[0,1], displacement[0,2]

        nodes = get_external_nodes()
        xy_u = []
        for node in nodes:
            (x, y, _) = node.center
            (ux, uy, uz) = calculate_displacement(node)
            if ~np.isnan(ux) & ~np.isnan(uy) & ~np.isnan(uz):
                xy_u.append([x, y, ux, uy, uz])
        return np.array(xy_u)


