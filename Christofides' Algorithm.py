#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:15:37 2020

@author: jinanwachikafavour
"""

#!pip install pytsp #Uncomment this line to install this useful module 
import itertools #module for working with iterables

import numpy as np #using numpy for efficient iteration of arrays
import networkx as nx

from networkx.algorithms.matching import max_weight_matching #modules that implements perfect matching
from networkx.algorithms.euler import eulerian_circuit

from pytsp.utils import minimal_spanning_tree #imports the MST class produced above


def christofides_tsp(graph, starting_node=0):
    """
    Function implements Christofides' algorithm beginning at
    a fixed point, which will be San Francisco by default.
        graph: 2d numpy array matrix where the index of the distances is the 
        position of the cities affected created in the dict below eg San Francisco at index 0,
        Seattle at index 1, LA index 2 etc. These cities are rep with their airport codes
        The distance from an airport to itself is always 0.
        The distance from any of the airports to the other is in miles and filled in the array.
        starting_node: of the TSP
    Returns:
        tour given by christofies TSP algorithm
    """

    mst = minimal_spanning_tree(graph, 'Prim', starting_node=0) #calls on MST function created above to the graph
    odd_degree_nodes = list(_get_odd_degree_vertices(mst)) #gets the odd vertices of the MST
    odd_degree_nodes_ix = np.ix_(odd_degree_nodes, odd_degree_nodes)
    nx_graph = nx.from_numpy_array(-1 * graph[odd_degree_nodes_ix])
    matching = max_weight_matching(nx_graph, maxcardinality=True) #implements perfect matching 
    euler_multigraph = nx.MultiGraph(mst) #produces an Euler graph with repeated edges
    for edge in matching:
        euler_multigraph.add_edge(odd_degree_nodes[edge[0]], odd_degree_nodes[edge[1]],
                                  weight=graph[odd_degree_nodes[edge[0]]][odd_degree_nodes[edge[1]]])
    euler_tour = list(eulerian_circuit(euler_multigraph, source=starting_node)) 
    path = list(itertools.chain.from_iterable(euler_tour)) #iterates through edges in graph
    return _remove_repeated_vertices(path, starting_node)[:-1]


def _get_odd_degree_vertices(graph):
    """
    Finds all the odd degree vertices in graph
    Args:
        graph: 2d np array as adj. matrix
    Returns:
    Set of vertices that have odd degree
    """
    odd_degree_vertices = set()
    for index, row in enumerate(graph):
        if len(np.nonzero(row)[0]) % 2 != 0: #if the vertice is odd
            odd_degree_vertices.add(index) #append to index
    return odd_degree_vertices
#airports in their order with respective distances from each other retrieved from Google maps
airports_and_distances = {'SFO': [0, 810.6, 381.5, 2472.3,1934.7, 2107.2, 2935.3,2047.9\
                                 ,3108.9,2135.9,3106.9,1265.1,2890.7,2816.1,2281.5],     
                         'SEA':[810.6,0,387.5,2478.2,1940.6,2094.5,14.2,2035.2,3096.2\
                               ,2123.1,3112.9,1252.3,2878.0,2803.4,2268.8],
                         'LAX':[381.5,387.5,0,2196.6,1576.1,1998.9,2826.9,1844.5\
                               ,3000.6,2027.5,2748.4,1055.0,2733.0,2671.0,2072.5],
                         'ATL':[2472.3,2478.2,2196.6,0,792.4,879.1,884.7,576.1,1086.4\
                               ,747.7,656.8,1405.4,775.1,660.6,552.3],
                         'IAH':[1934.7,1940.6,1576.1,792.4,0,1123.8,1655.8,775.8,1852.2\
                               ,1081.5,1197.9,1005.5,1546.2,1404.8,989.2],
                         'MSN':[2107.2,2094.5,1998.9,879.1,1123.8,0,2926.9,308,963,109,1301,826,764\
                               ,684,284],
                         'JFK':[2935.3,14.2,2826.9,884.7,1655.8,2926.9,0,989.1,214.2,826.5,1297.9\
                               ,1792.7,113.9,260.2,751.6],
                         'STL':[2047.9,2035.2,1844.5,576.1,775.8,308,989.1,0,1193.9,297.9,1216.8\
                               ,842.8,901.5,818.1,241.0],
                         'BOS':[3108.9,3096.2,3000.6,1086.4,1852.2,963,214.2,1193.9,0,2122.6\
                               ,3112.3,1251.8,2877.4,2802.8,2268.2],
                         'ORD':[2135.9,2123.1,2027.5,747.7,1081.5,109,826.5,297.9,2122.6,0\
                               ,1399.3,992.1,782.4,707.8,207.6],
                         'MIA':[3106.9,3112.9,2748.4,656.8,1197.9,1301,1297.9,1216.8,3112.3,\
                               1399.3,0,2059.8,1180.1,1065.6,1206.8],
                         'DEN':[1265.1,1252.3,1055.0,1405.4,1005.5,826,1792.7,842.8,1251.8,\
                               992.1,2059.8,0,2879.7,2805.1,2270.5],
                         'PHL':[2890.7,2878.0,2733.0,775.1,1546.2,764,113.9,901.5,2877.4,\
                               782.4,1180.1,2879.7,0,150.3,664.1],
                         'IAD':[2816.1,2803.4,2671.0,660.6,1404.8,684,260.2,818.1,2802.8,\
                               707.8,1065.6,2805.1,150.3,0,851.6],
                         'IND':[2281.5,2268.8,2072.5,552.3,989.2,284,751.7,241.0,2268.2,\
                               207.4,1206.8,2270.5,664.1,851.6,0]}
def _remove_repeated_vertices(path, starting_node):
    """Function removes the repeated vertices by short-cutting the euler path created initially"""
    path = list(dict.fromkeys(path).keys()) #extracts the keys from dictionary which rep the weighted edges
    path.append(starting_node) #adds these edges to the starting city
    return path #returns a minimum/efficient path
graph =  np.array([[0, 810.6, 381.5, 2472.3,1934.7, 2107.2, 2935.3,2047.9\
                                 ,3108.9,2135.9,3106.9,1265.1,2890.7,2816.1,2281.5],
                   [810.6,0,387.5,2478.2,1940.6,2094.5,14.2,2035.2,3096.2\
                               ,2123.1,3112.9,1252.3,2878.0,2803.4,2268.8],
                  [381.5,387.5,0,2196.6,1576.1,1998.9,2826.9,1844.5\
                               ,3000.6,2027.5,2748.4,1055.0,2733.0,2671.0,2072.5],
                  [2472.3,2478.2,2196.6,0,792.4,879.1,884.7,576.1,1086.4\
                               ,747.7,656.8,1405.4,775.1,660.6,552.3],
                  [1934.7,1940.6,1576.1,792.4,0,1123.8,1655.8,775.8,1852.2\
                               ,1081.5,1197.9,1005.5,1546.2,1404.8,989.2]
                  ,[2107.2,2094.5,1998.9,879.1,1123.8,0,2926.9,308,963,109,1301,826,764\
                               ,684,284],
                  [2935.3,14.2,2826.9,884.7,1655.8,2926.9,0,989.1,214.2,826.5,1297.9\
                               ,1792.7,113.9,260.2,751.6],
                  [2047.9,2035.2,1844.5,576.1,775.8,308,989.1,0,1193.9,297.9,1216.8\
                               ,842.8,901.5,818.1,241.0]
                  ,[3108.9,3096.2,3000.6,1086.4,1852.2,963,214.2,1193.9,0,2122.6\
                               ,3112.3,1251.8,2877.4,2802.8,2268.2],
                  [2135.9,2123.1,2027.5,747.7,1081.5,109,826.5,297.9,2122.6,0\
                               ,1399.3,992.1,782.4,707.8,207.6],
                  [3106.9,3112.9,2748.4,656.8,1197.9,1301,1297.9,1216.8,3112.3,\
                               1399.3,0,2059.8,1180.1,1065.6,1206.8],
                  [1265.1,1252.3,1055.0,1405.4,1005.5,826,1792.7,842.8,1251.8,\
                               992.1,2059.8,0,2879.7,2805.1,2270.5],
                  [2890.7,2878.0,2733.0,775.1,1546.2,764,113.9,901.5,2877.4,\
                               782.4,1180.1,2879.7,0,150.3,664.1],
                  [2816.1,2803.4,2671.0,660.6,1404.8,684,260.2,818.1,2802.8,\
                               707.8,1065.6,2805.1,150.3,0,851.6],
                  [2281.5,2268.8,2072.5,552.3,989.2,284,751.7,241.0,2268.2,\
                               207.4,1206.8,2270.5,664.1,851.6,0]])

                                      
christofides_tsp(graph) #the output is the position of the cities in the 2-d numpy array.
#to produce a more useful and understandable output, I typed the order of the cities myself by
# referencing the output of Christofides' algorithm then summed the distances in this order by myself


print('The desired route is SFO-DEN-MSN-ORD-IND-STL-IAH-ATL-MIA-IAD-PHL-JFK-BOS-SEA-LAX''\n'
     'This represents the cities San Francisco, Denver,Madison,Chicago,Indianapolis,St.Louis,'\
     'Houston,Atlanta,''\n' 'Miami,Washington DC, Philadelphia,New York,Boston, Seattle and Los Angeles' \
     'respectively')
print('The total distance is', 1265.1+2096.2+135+207.6+241.0+773.1+793.1+656.8+1065.6+151.3+115.7+241.2+793.7\
     +387.5,'miles')