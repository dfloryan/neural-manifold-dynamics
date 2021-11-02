#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from overlap import overlap
from chartMap import autoencoder
import keras
import tensorflow as tf

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

# Generate torus data
r1 = 1.0
r2 = 0.5
nSamples = 100
theta = np.linspace(0, 2*np.pi, nSamples, endpoint = False)
phi = 3.0*theta
X = np.vstack(((r1 + r2*np.cos(phi))*np.cos(theta), (r1 + r2*np.cos(phi))*np.sin(theta), r2*np.sin(phi))).transpose()

# Create k-NN graph
nNeighbors = 2
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
for i in range(2):
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
fig = plt.figure()
ax = p3.Axes3D(fig)

print("Compute maps for each cluster...")
chartMaps = np.empty((nClus, ), dtype = object)
chartDyn = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    chartMaps[i] = autoencoder(nDim)
    enStruct = [32, 32, 16, 4, nDim]
    enAct = ['elu', 'elu', 'elu', 'elu', None]
    deStruct = [4, 16, 32, 32, X.shape[1]]
    deAct = ['elu', 'elu', 'elu', 'elu', None]
    chartMaps[i].build(X.shape[1], enStruct, enAct, deStruct, deAct)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
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
    trainArgs = {'epochs': 1000, 'batch_size': X[clusPts[i], :].shape[0] - 1}
    ind = clusPts[i][np.nonzero(np.diff(clusPts[i]) == 1)] # indices for first snapshots in snapshot pairs
    chartDyn[i] = nnDyn(nDim, nnStruct, nnAct, chartMaps[i].encode(X[ind, :]), chartMaps[i].encode(X[ind + 1, :]), optArgs, trainArgs)
    
    # Calculate and plot reconstruction of data
    Xrecon = chartMaps[i].decode(chartMaps[i].encode(X[clusPts[i], :]))
    ax.scatter(Xrecon[:, 0], Xrecon[:, 1], Xrecon[:, 2],
            facecolor = plt.cm.Set1(i), marker = 'x', zorder = 3)
    
    # Plot line from reconstruction to original point
    xx = np.concatenate((Xrecon[:, 0:1], X[clusPts[i],:][:, 0:1]), axis = 1)
    yy = np.concatenate((Xrecon[:, 1:2], X[clusPts[i],:][:, 1:2]), axis = 1)
    zz = np.concatenate((Xrecon[:, 2:3], X[clusPts[i],:][:, 2:3]), axis = 1)
    for j in range(xx.shape[0]):
        ax.plot(xx[j, :], yy[j, :], zz[j, :], color = 'k', linewidth = 0.5, zorder = 1)
    
    # Plot reconstruction of entire torus in this chart
    theta = np.linspace(0, 2*np.pi, 1000, endpoint = False)
    phi = 3.0*theta
    Xtorus = np.vstack(((r1 + r2*np.cos(phi))*np.cos(theta), (r1 + r2*np.cos(phi))*np.sin(theta), r2*np.sin(phi))).transpose()
    Xrecon = chartMaps[i].decode(chartMaps[i].encode(Xtorus))
    ax.plot(Xrecon[:, 0], Xrecon[:, 1], Xrecon[:, 2], color = plt.cm.Set1(i), zorder = 3)
    
print("Done")

# Store encoded versions of all points
Xencode = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    Xencode[i] = chartMaps[i].encode(X[clusPts[i], :])

# Plot result
xx = np.concatenate((X[edges[0], 0:1], X[edges[1], 0:1]), axis = 1)
yy = np.concatenate((X[edges[0], 1:2], X[edges[1], 1:2]), axis = 1)
zz = np.concatenate((X[edges[0], 2:3], X[edges[1], 2:3]), axis = 1)

# Plot edges of graph
for i in range(xx.shape[0]):
    ax.plot(xx[i, :], yy[i, :], zz[i, :], color = (0.8, 0.8, 0.8), linewidth = 0.5, zorder = -nClus)

# Plot nodes of each cluster
for i in range(nSamples):
    for j in range(1, len(clus[i])):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2],
                facecolor = plt.cm.Set1(clus[i][j]),
                s = j*j*100, edgecolor = plt.cm.Set1(clus[i][j]), zorder = -j)

for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                facecolor = plt.cm.Set1(l),
                s = 20, edgecolor = plt.cm.Set1(l), zorder = 1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Dynamics part
# Evolve an initial condition forward in time along the torus using the 
# charts and dynamics on them
x0 = X[0:1, :]

# Find which cluster the point is in initially, map into local coordinates
clusNew = kmeans.predict(x0)[0]
y = chartMaps[clusNew].encode(x0)

# Evolve the point forward in time
nsteps = 1000
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

# Plot the trajectory
fig = plt.figure()
ax = p3.Axes3D(fig)
for i in np.unique(clusArr):
    ax.scatter(xArr[np.squeeze(clusArr == i), 0], xArr[np.squeeze(clusArr == i), 1], xArr[np.squeeze(clusArr == i), 2],
                facecolor = plt.cm.Set1(i),
                s = 20, edgecolor = plt.cm.Set1(i), zorder = 1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig, ax = plt.subplots()
x = np.arange(nsteps + 1)
for i in np.unique(clusArr):
    ax.scatter(x[np.squeeze(clusArr == i)], xArr[np.squeeze(clusArr == i), 0],
                facecolor = plt.cm.Set1(i),
                s = 20, linewidth = 1, zorder = 2)
    ax.scatter(x[np.squeeze(clusArr == i)], xArr[np.squeeze(clusArr == i), 1],
                facecolor = plt.cm.Set1(i), marker = 'v',
                s = 20, linewidth = 1, zorder = 2)
    ax.scatter(x[np.squeeze(clusArr == i)], xArr[np.squeeze(clusArr == i), 2],
                facecolor = plt.cm.Set1(i), marker = '^',
                s = 20, linewidth = 1, zorder = 2)

theta = np.linspace(0, 2*np.pi*(nsteps + 1)/nSamples, nsteps + 1, endpoint = False)
phi = 3.0*theta
X = np.vstack(((r1 + r2*np.cos(phi))*np.cos(theta), (r1 + r2*np.cos(phi))*np.sin(theta), r2*np.sin(phi))).transpose()
ax.plot(X[:, 0], color = (0.8, 0.8, 0.8), zorder = 1)
ax.plot(X[:, 1], color = (0.8, 0.8, 0.8), zorder = 1)
ax.plot(X[:, 2], color = (0.8, 0.8, 0.8), zorder = 1)

ax.set_xlabel('time step')
ax.set_ylabel('x, y, z')
