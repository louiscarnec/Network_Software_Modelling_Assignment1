#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:28:44 2017

@author: Carnec
"""

import numpy as np
import networkx as nx
from heapq import heapify, heappush, heappop
import math
import time
import matplotlib.pyplot as plt


""" Functions to create randomly generated graphs of different sizes """


def RCUEWG(n): #from tutorial solutions 
    """RCUEWG stands for random complete undirected
    edge-weighted graph, obviously."""
    M = np.random.random((n, n))
    G = nx.Graph()
    for i in range(n):
        # notice we are using the upper triangle only
        # that is we discard M_ii and M_ji where j<i
        for j in range(i+1, n):
            G.add_edge(i, j, weight=M[i, j])
    return G


def RandUWGraph(n): #This function was used in testing  
    G=nx.dense_gnm_random_graph(n,n) #creates a dense graph of n nodes and n edges
    for (u,v,w) in G.edges(data=True):
        w['weight'] = np.random.random() #Non-negative weights from a uniform dist uni[0.0,1.0)
    return G
    
#G5 = RandUWGraph(5)
#print("G_edges",G5.edges(data=True))
#G100 = RandUWGraph(100)
#print("G_edges",G100.edges(data=True))

""" Bidirectional dijkstra is between source and destination node, thus we are
looking at a specific example of Dijkstra applied between two points.
In doing so we need to pick two points and assign them to be source and destination """


def source_dest_nodes(n): #This function randomly picks a source and destination node
    sourcenode = 0
    destnode = 0
    while sourcenode == destnode:
        sourcenode = np.random.randint(n)
        destnode = np.random.randint(n) 
    return sourcenode, destnode
    
    
""" Dijkstra Algorithm using binary heap for S """

def dijkstraHeapTarget(G, r, target):
    if not G.has_node(r): #raise value error is source node does not exist
        raise ValueError("Source node " + str(r) + " not present")
    
    if not G.has_node(target): #raise value error is target node does not exist
        raise ValueError("Target node " + str(target) + " not present") 

    for e1, e2, d in G.edges(data=True): #weights must be non-negative
        if d["weight"] < 0:
            raise ValueError("Negative weight on edge " + str(e1) + "-" + str(e2))

    P = {r} # permanent set
    S = [] # V-P. This is the crucial data structure.
    D = {} # estimates of SPST lengths
    p = {} # parent-pointers
#    A = set() # our result, an SPST
    

    for n in G.nodes(): #we initialise the distances for enighbours of s
        if n == r:
            D[n] = 0
        else:
            if G.has_edge(r, n):
                D[n] = G[r][n]["weight"]
            else:
                D[n] = math.inf #if not neighbour of s, have infinity as distance

            p[n] = r
            heappush(S,(D[n], n)) #push scanned nodes to the heap S

    while len(S): #while the queue S is not empty scan
        Du,u = heappop(S) #pop out node with lowest weight to scan
        if u == target: #if scanned node is the target, terminate
            break
        if u in P: continue #if u in P, have already seen the node, no need to scan it

        P.add(u) # move one item to the permanent set P and to SPST A
#        A.add((p[u], u))

        for Dv, v in S: #update the distance of neighbours of u
            if v in P: continue #if neighbour of u is already permanent, we do not change distance
            if G.has_edge(u, v):
                if D[v] > D[u] + G[u][v]["weight"]: # if lower distance found from s to v, update distance
                    D[v] = D[u] + G[u][v]["weight"]
                    p[v] = u
                    heappush(S,(D[v],v)) # v is added to the priority queue S, or distance of v is updated
    
    #Once terminated, we use parent pointers to find SP                            
    end = target
    path = [end]
    while end != r:
        end = p[end]
        path.append(end)
    path.reverse() #because path in reverse need to reverse it
    
    if D[target] == math.inf: #if the distance to target is infinity, no path could be found between s and t. 
        raise nx.NetworkXNoPath("No path between %s and %s." % (source, target)) #Raise Error taken from Networkx Implementaion
        
    
    return path, D[target] # let's return the SP to target and distance to target



""" Bidirectional Dijkstra Algorithm"""

def bidirectionalDijkstra(G, source, target):
    if not G.has_node(source):
        raise ValueError("Source node " + str(source) + " not present")
        
    if not G.has_node(target):
        raise ValueError("Target node " + str(target) + " not present")    

    for e1, e2, d in G.edges(data=True):
        if d["weight"] < 0:
            raise ValueError("Negative weight on edge " + str(e1) + "-" + str(e2))
    
    P = [{source},{target}] # permanent set
    S = [[],[]] # V-P. This is the crucial data structure.
    D = [{},{}] # estimates of SPST lengths
    p = [{},{}] # parent-pointers
             
   
# Initialise the heap forward and backward heaps
#forward heap is S[0] and backward heap is S[1]
    for n in G.nodes():
        if n == source:
            D[0][n] = 0
        else:
            if G.has_edge(source, n):
                D[0][n] = G[source][n]["weight"]
            else:
                D[0][n] = math.inf

            p[0][n] = source
            heappush(S[0],(D[0][n], n))     

    for n in G.nodes():
        if n == target:
            D[1][n] = 0
        else:
            if G.has_edge(n, target):
                D[1][n] = G[n][target]["weight"]
            else:
                D[1][n] = math.inf

            p[1][n] = target
            heappush(S[1],(D[1][n], n))   

    finaldist = 1e3333333333 #set finaldist (to be returned as distance from s to t)
                                #to very large number so that finaldist is updated if 
                                #the distance found is by algorithm is shorter
    direction = 1 

 
    while len(S[0]) and len(S[1]): #while neither forward or backward heaps are empty, do
        #start with forward direction (direction = 0)
        direction = 1 - direction #changes from forward (direction=0) to backward (direction = 1) at each iteration
        
        Du,u = heappop(S[direction]) #pop priority node from relevant heap direction
        
        if u in P[direction]: #if u has already been seen, next iteration
            continue
        
        P[direction].add(u)
#        A[direction].add(p[u],u)
        
        if u in P[1 - direction]: #if scanned u is already in permanent set of other direction
                                    #(seen in other direction, terminate the while loop)
                                    #however, we have shown that the node the algorithm
                                    #terminated on is not necessarily on the shortest path.
                                    #need to look for node x on SP:
            for x in D[0] and D[1]: #look for the node x in distances with minimum forward and backward distance sum
                totaldist = D[0][x] + D[1][x]
                if totaldist < finaldist:
                    finaldist = totaldist 
                    X = x
            
            end = X #like unidirectional algorithm, use parent pointers to find path
            path = [end] #first create list of parent pointers for forward search
            while end != source:
                end = p[0][end]
                path.append(end)
            path.reverse() #reverse path of forward search to get path form s to x
            
            end = X
            while end != target: #add parents for backward search
                end = p[1][end]
                path.append(end)
            finalpath = path
        
            return (finalpath, finaldist) #return SP and distance
                
        for Dv, v in S[direction]: #add v / update distance for neighbours of u in given direction
            if v in P[direction]: continue
            if G.has_edge(u, v):
                if D[direction][v] > D[direction][u] + G[u][v]["weight"]:
                    D[direction][v] = D[direction][u] + G[u][v]["weight"]
                    p[direction][v] = u
                    heappush(S[direction],(D[direction][v],v))
 
                
            
if __name__== "__main__":
    
    listofn = [5,10,20,50]#,100,1000,10000]

    timesDictDijkstra = {} #runtime dictionary for Dijkstra
    timesDictBiDijkstra = {} #runtime dictionary for bidirectional Dijkstra

    for n in listofn:   
        G = RandUWGraph(n)
    #    G = RCUEWG(n)
        print('--------')
        print('--------')
        print('--------')
        print('Graph of size ',n)
        source, dest = source_dest_nodes(n)
        print('Random source node:', source)
        print('Random destination node:', dest)
        print("---")
        print("Networkx dijkstra length:", nx.dijkstra_path_length(G,source,dest))
        print("Networkx dijkstra path:",nx.dijkstra_path(G,source,dest))
        bilength,bipath=nx.bidirectional_dijkstra(G,source,dest)
        print("Networkx bidirectional dijkstra length:", bilength)
        print("Networkx bidirectional dijkstra path:", bipath)
        print("---")
        implemDijkstraPath, implemDijkstraLength = dijkstraHeapTarget(G, source, dest)
        implemBiDijkstraPath, implemBiDijkstraLength = bidirectionalDijkstra(G, source, dest)
        print("Implemented dijkstra length:", implemDijkstraLength)
        print("Implemented dijkstra path:", implemDijkstraPath)
        print("---")        
        print("Implemented bidirectional dijkstra length:", implemBiDijkstraLength)
        print("Implemented bidirectional dijkstra path:", implemBiDijkstraPath)
#
#        line1 = plt.plot(*zip(*sorted(timesDictBiDijkstra.items())),label = "Bidirectional")
#        line2 = plt.plot(*zip(*sorted(timesDictDijkstra.items())),label = "Dijkstra")
#        plt.ylabel('Time')
#        plt.xlabel('Graph Size n')
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
#        plt.title("Runtime")
#        plt.savefig('/Users/Carnec/Desktop/Business_Analytics/networksoftwaremodelling/NSM_assignment1/runtime1.png')
#        plt.show()
#        
        
        
        
        
        
        
        