#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERLAP: overlap between sets of vertices. Expand all sets out by one
neighbour. 
 
   OVERLAP(E,C,CPTS) returns assignments of vertices to clusters so that all 
   clusters expand out by one neighbour. 
 
   INPUTS
       e           adjacency list of undirected edges in a graph (array of 
                   length V, e[i] is a list of vertices connected to vertex i)
       c           assignment of vertices to cluster (array of length V, c[i] 
                   is a list of clusters that vertex i belongs to), modified
                   by this function
       cPts        assignment of vertices to cluster (array of length C, 
                   cPts[i] is a list of vertices that belong to cluster i), 
                   modified by this function
 
   OUTPUTS
       none

   NOTES
       vertices are assumed to be indexed by contiguous integers starting
       from 0
       clusters are assumed to be indexed by contiguous integers starting
       from 0


   FLORYAN, DANIEL
   September 22, 2020
   Edited September 24, 2020
"""

import numpy as np
from collections import deque

def overlap(e, c, cPts):
    # Loop through each cluster
    for i in range(len(cPts)):
        # Perform depth-first search on current cluster
        marked = np.zeros((len(c), ), dtype = bool)
        for v in cPts[i]:
            if not marked[v]:
                # can use bfs or dfs here
                # dfs may go past python recursion depth limit
                bfs(e, c, cPts, marked, v, i)

def dfs(e, c, cPts, marked, v, clusInd):
    marked[v] = True
    if v in cPts[clusInd]:
        for w in e[v]:
            if not marked[w]:
                dfs(e, c, cPts, marked, w, clusInd)
    else:
        cPts[clusInd].append(v)
        c[v].append(clusInd)

def bfs(e, c, cPts, marked, v, clusInd):
    marked[v] = True
    queue = deque()
    queue.append(v)
    cPtsSet = set(cPts[clusInd])
    while len(queue) > 0: # while queue not empty
        s = queue.popleft()
        if s in cPtsSet:
            for w in e[s]:
                if not marked[w]:
                    marked[w] = True
                    queue.append(w)
        else:
            cPts[clusInd].append(s)
            c[s].append(clusInd)
