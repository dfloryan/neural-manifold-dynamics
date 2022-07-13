#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from overlap import overlap
from chartMap import autoencoder
import keras
import tensorflow as tf
import scipy.io

def nnDyn(n, nnStruct, nnAct, X0, X1, optArgs = {}, trainArgs = {}):
    # Set precision
    keras.backend.set_floatx('float64')
    
    # Assemble neural network
    nnInput = keras.Input(shape = (n,))
    nnState = nnInput
    for i in range(len(nnStruct)):
        nnState = keras.layers.Dense(nnStruct[i], activation = nnAct[i])(nnState)
    
    # Build neural network
    nnEvolve = keras.Model(inputs = nnInput, outputs = nnState)
    
    # Train the neural network
    nnEvolve.compile(**optArgs)
    nnEvolve.fit(X0, X1, **trainArgs)
    return nnEvolve

# Load reaction diffusion data
nSamples = 200
mat = scipy.io.loadmat('reactDifNeumannData.mat')
X = mat['z']
X = X[5000:5200, :] # grab roughly one period of data

# Create k-NN graph
nNeighbors = 4
connectivity = kneighbors_graph(X, n_neighbors = nNeighbors, include_self = False)
edges = connectivity.nonzero()

# Create adjacency list. The graph resulting from k-NN is a directed graph, 
# but we will make it undirected by removing the direction of all edges. We 
# will then remove any redundant edges. 
adjList = np.empty((nSamples, ), dtype = object)
for i in range(nSamples):
    adjList[i] = []
for i in range(len(edges[0])):
    adjList[edges[0][i]].append(edges[1][i])
    adjList[edges[1][i]].append(edges[0][i])
for i in range(nSamples):
    adjList[i] = list(set(adjList[i]))

# Compute clustering
print("Compute k-means clustering...")
nClus = 3
kmeans = KMeans(n_clusters = nClus, random_state = 0).fit(X)
label = kmeans.labels_
print("Done")

# Create array of lists, one list for each data point, each list containing 
# the cluster indices that point belongs to
clus = np.empty((nSamples, ), dtype = object)
for i in range(nSamples):
    clus[i] = [label[i]]

# Create array of lists, one list for each cluster, each list containing the
# data point indices that cluster contains
clusPts = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    clusPts[i] = np.nonzero(label == i)[0].tolist()

# Make clusters overlap
print("Compute cluster overlap...")
for i in range(1):
    overlap(adjList, clus, clusPts)
print("Done")

# Convert clusPts to array of arrays instead of array of lists. Sort each array.
print("Sort points in clusters...")
for i in range(nClus):
    clusPts[i] = np.array(clusPts[i])
    clusPts[i].sort()
print("Done")

# Find the mapping for each chart
nDim = 1 # dimension to reduce to

print("Compute maps for each cluster...")
chartMaps = np.empty((nClus, ), dtype = object)
chartDyn = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    chartMaps[i] = autoencoder(nDim)
    enStruct = [128, 64, 16, 8, nDim]
    enAct = ['elu', 'elu', 'elu', 'elu', None]
    deStruct = [8, 16, 64, 128, X.shape[1]]
    deAct = ['elu', 'elu', 'elu', 'elu', None]
    chartMaps[i].build(X.shape[1], enStruct, enAct, deStruct, deAct)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.8, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 1000, 'batch_size': X[clusPts[i], :].shape[0]}
    chartMaps[i].train(X[clusPts[i], :], optArgs, trainArgs)
    
    # Find the dynamics
    nnStruct = [32, 32, 16, 4, nDim]
    nnAct = ['elu', 'elu', 'elu', 'elu', None]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 2000, 'batch_size': X[clusPts[i], :].shape[0] - 1}
    ind = clusPts[i][np.nonzero(np.diff(clusPts[i]) == 1)] # indices for first snapshots in snapshot pairs
    chartDyn[i] = nnDyn(nDim, nnStruct, nnAct, chartMaps[i].encode(X[ind, :]), chartMaps[i].encode(X[ind + 1, :]), optArgs, trainArgs)

print("Done")

# Store encoded versions of all points
Xencode = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    Xencode[i] = chartMaps[i].encode(X[clusPts[i], :])
    
# Plot original data, its reconstruction, and difference between the two
fig, ax = plt.subplots(1, 3)
c = ax[0].contourf(mat['x'], mat['y'], np.reshape(X[0, 0:101**2], (101, 101)), levels = np.linspace(-1, 1, 21), cmap = 'PuOr_r')
ax[0].set_title('Data')
fig.colorbar(c, ax = ax[0], ticks = [-1, 0, 1])
Xrecon = chartMaps[clus[0][0]].decode(chartMaps[clus[0][0]].encode(X[0:1,:]))
c = ax[1].contourf(mat['x'], mat['y'], np.reshape(Xrecon[:, 0:101**2], (101, 101)), levels = np.linspace(-1, 1, 21), cmap = 'PuOr_r')
ax[1].set_title('Decode(Encode(Data))')
fig.colorbar(c, ax = ax[1], ticks = [-1, 0, 1])
c = ax[2].contourf(mat['x'], mat['y'], np.reshape(X[0, 0:101**2] - Xrecon[:, 0:101**2], (101, 101)), cmap = 'RdBu_r')
ax[2].set_title('Error')
fig.colorbar(c, ax = ax[2])

# Dynamics part
# Evolve an initial condition forward in time along the limit cycle using the 
# charts and dynamics on them
x0 = X[0:1, :]

# Find which cluster the point is in initially, map into local coordinates
clusNew = kmeans.predict(x0)[0]
y = chartMaps[clusNew].encode(x0)

# Evolve the point forward in time
nsteps = 5000
yArr = np.zeros((nsteps + 1, nDim))
yArr[0, :] = y
xArr = np.zeros((nsteps + 1, X.shape[1]))
xArr[0, :] = x0
clusArr = np.zeros((nsteps + 1, 1), dtype = int)
clusArr[0, 0] = clusNew
for i in range(nsteps):
    # Map points forward
    y = chartDyn[clusNew].predict(y)
    
    # Find nearest training point in chart. Switch charts if necessary.
    clusOld = clusNew
    dist2 = np.sum((Xencode[clusNew] - y)**2, axis = 1)
    clusNew = clus[clusPts[clusNew][np.argmin(dist2)]][0]
    if clusNew != clusOld:
        y = chartMaps[clusNew].encode(chartMaps[clusOld].decode(y))
    
    # Store trajectory
    yArr[i + 1, :] = y
    xArr[i + 1, :] = chartMaps[clusNew].decode(y)
    clusArr[i + 1, 0] = clusNew

# Plot the last snapshot, compare to ground truth
fig, ax = plt.subplots(1, 3)
c = ax[0].contourf(mat['x'], mat['y'], np.reshape(mat['z'][5000 + nsteps, 0:101**2], (101, 101)), levels = np.linspace(-1, 1, 21), cmap = 'PuOr_r')
ax[0].set_title('Ground truth')
fig.colorbar(c, ax = ax[0], ticks = [-1, 0, 1])
c = ax[1].contourf(mat['x'], mat['y'], np.reshape(xArr[nsteps, 0:101**2], (101, 101)), levels = np.linspace(-1, 1, 21), cmap = 'PuOr_r')
ax[1].set_title('Predicted')
fig.colorbar(c, ax = ax[1], ticks = [-1, 0, 1])
c = ax[2].contourf(mat['x'], mat['y'], np.reshape(mat['z'][5000 + nsteps, 0:101**2] - xArr[nsteps, 0:101**2], (101, 101)), cmap = 'RdBu_r')
ax[2].set_title('Error')
fig.colorbar(c, ax = ax[2])
