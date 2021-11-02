#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from overlap import overlap
from chartMap import hybridnn
import keras
import tensorflow as tf
import scipy.io
from scipy import linalg
import mpl_toolkits.mplot3d.axes3d as p3

def nnDyn(n, nnStruct, nnAct, X, X0, X1, optArgs = {}, trainArgs = {}):
    # Set precision
    keras.backend.set_floatx('float64')
    
    # Get PCA change-of-basis matrix and mean
    Xmean = np.mean(X, axis = 0)
    U, s, _ = linalg.svd((X - Xmean).transpose(), full_matrices = True)
    U = keras.backend.variable(value = U)
    s = keras.backend.variable(value = s)
    Xmean = keras.backend.variable(value = Xmean)
    
    # Assemble neural network
    nnInput = keras.Input(shape=(n,))
    nnState = nnInput

    # Change to PCA coordinates, normalize each coordinate axis
    nnState = keras.layers.Lambda(lambda x: x - Xmean)(nnState)
    nnState = keras.layers.Lambda(lambda x: tf.einsum("ij,jk->ik", x, U))(nnState)
    nnState = keras.layers.Lambda(lambda x: x/s*np.sqrt(X.shape[0]))(nnState)
        
    # Put the state through a neural network
    for i in range(len(nnStruct)):
        nnState = keras.layers.Dense(nnStruct[i], activation = nnAct[i])(nnState)
        
    # Change back to original un-normalized coordinates
    nnState = keras.layers.Lambda(lambda x: x*s/np.sqrt(X.shape[0]))(nnState)
    nnState = keras.layers.Lambda(lambda x: tf.einsum("ij,kj->ik", x, U))(nnState)
    nnState = keras.layers.Lambda(lambda x: x + Xmean)(nnState)
    
    # Build neural network
    nnEvolve = keras.Model(inputs = nnInput, outputs = nnState)
    
    # Train the neural network
    nnEvolve.compile(**optArgs)
    nnEvolve.fit(X0, X1, **trainArgs)
    return nnEvolve

# Load bursting data
nSamples = 6565
mat = scipy.io.loadmat('ksdataBurstingBalanced.mat')
X = mat['udata'].transpose()
t = np.squeeze(mat['t'])

# Load bursting data for dynamics part
nDyn = 2189*21
mat = scipy.io.loadmat('ksdataBurstingDynamics.mat')
Xdyn = mat['udata'].transpose()
tdyn = np.squeeze(mat['t'])

# Create k-NN graph
nNeighbors = 4
connectivity = kneighbors_graph(X, n_neighbors = nNeighbors, include_self = False)
edges = connectivity.nonzero()

connectivityDyn = kneighbors_graph(Xdyn, n_neighbors = nNeighbors, include_self = False)
edgesDyn = connectivityDyn.nonzero()

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
    
adjListDyn = np.empty((nDyn, ), dtype = object)
for i in range(nDyn):
    adjListDyn[i] = []
for i in range(len(edgesDyn[0])):
    adjListDyn[edgesDyn[0][i]].append(edgesDyn[1][i])
    adjListDyn[edgesDyn[1][i]].append(edgesDyn[0][i])
for i in range(nDyn):
    adjListDyn[i] = list(set(adjListDyn[i]))

# Compute clustering
print("Compute k-means clustering...")
nClus = 6
kmeans = KMeans(n_clusters = nClus, random_state = 0).fit(X)
label = kmeans.labels_
labelDyn = kmeans.predict(Xdyn)
print("Done")

# Create array of lists, one list for each data point, each list containing 
# the cluster indices that point belongs to
clus = np.empty((nSamples, ), dtype = object)
for i in range(nSamples):
    clus[i] = [label[i]]
    
clusDyn = np.empty((nDyn, ), dtype = object)
for i in range(nDyn):
    clusDyn[i] = [labelDyn[i]]

# Create array of lists, one list for each cluster, each list containing the
# data point indices that cluster contains
clusPts = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    clusPts[i] = np.nonzero(label == i)[0].tolist()
    
clusPtsDyn = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    clusPtsDyn[i] = np.nonzero(labelDyn == i)[0].tolist()

# Make clusters overlap
print("Compute cluster overlap...")
for i in range(2):
    overlap(adjList, clus, clusPts)
print("Done")

print("Compute cluster overlap...")
for i in range(2):
    overlap(adjListDyn, clusDyn, clusPtsDyn)
print("Done")

# Convert clusPts to array of arrays instead of array of lists. Sort each array.
print("Sort points in clusters...")
for i in range(nClus):
    clusPts[i] = np.array(clusPts[i])
    clusPts[i].sort()
print("Done")

print("Sort points in clusters...")
for i in range(nClus):
    clusPtsDyn[i] = np.array(clusPtsDyn[i])
    clusPtsDyn[i].sort()
print("Done")

# Find the mapping for each chart
nDim = 3 # dimension to reduce to

print("Compute maps for each cluster...")
chartMaps = np.empty((nClus, ), dtype = object)
chartDyn = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    chartMaps[i] = hybridnn(nDim)
    chartMaps[i].trainPCA(X[clusPts[i], :])
    enStruct = [128, 64, 16, 8, nDim]
    enAct = ['elu', 'elu', 'elu', 'elu', None]
    deStruct = [8, 16, 64, 128, X.shape[1]]
    deAct = ['elu', 'elu', 'elu', 'elu', None]
    chartMaps[i].build(X.shape[1], enStruct, enAct, deStruct, deAct, 1.0)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.8, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 1000, 'batch_size': X[clusPts[i], :].shape[0]}
    chartMaps[i].train(X[clusPts[i], :], optArgs, trainArgs)
    
    # Find the dynamics
    nnStruct = [32, 32, 16, 4, nDim]
    nnStruct = [16, 64, 64, 16, nDim]
    nnAct = ['elu', 'elu', 'elu', 'elu', None]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 1000, 'batch_size': Xdyn[clusPtsDyn[i], :].shape[0] - 1}
    ind = clusPtsDyn[i][np.nonzero((np.diff(clusPtsDyn[i]) == 1) & (np.diff(tdyn[clusPtsDyn[i]]) > 0))] # indices for first snapshots in snapshot pairs
    chartDyn[i] = nnDyn(nDim, nnStruct, nnAct, chartMaps[i].encode(Xdyn[clusPtsDyn[i], :]), chartMaps[i].encode(Xdyn[ind, :]), chartMaps[i].encode(Xdyn[ind + 1, :]), optArgs, trainArgs)

print("Done")

# Store encoded versions of all points
Xencode = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    Xencode[i] = chartMaps[i].encode(X[clusPts[i], :])

# Plot data in Fourier space
Xhat = np.fft.fft(X)
fig = plt.figure()
ax = p3.Axes3D(fig)

# Plot skeleton of attractor with faint line
ax.plot(np.real(Xhat[:, 2]), np.imag(Xhat[:, 1]), np.real(Xhat[:, 1]),
               color = (0.8, 0.8, 0.8), linewidth = 0.5, zorder = -nClus)

# Plot randomly sampled points from dataset
fs = 2 # plot every fs points
for i in range(nClus):
    Xtemp = Xhat[label == i, :]
    Xtemp = Xtemp[np.random.randint(0, len(Xtemp), int(len(Xtemp)/fs)), :]
    ax.scatter(np.real(Xtemp[:, 2]), np.imag(Xtemp[:, 1]), np.real(Xtemp[:, 1]),
               facecolor = plt.cm.Set1(i),
                s = 5, edgecolor = plt.cm.Set1(i), zorder = 1)

# Plot all points resulting from overlap
for i in range(nSamples):
    for j in range(len(clus[i]) - 1, 0, -1):
        ax.scatter(np.real(Xhat[i,2]), np.imag(Xhat[i,1]), np.real(Xhat[i,1]),
                edgecolor = plt.cm.Set1(clus[i][j]), facecolor = (0, 0, 0, 0),
                s = j*j*40, linewidth = 2, zorder = -j)
        ax.scatter(np.real(Xhat[i,2]), np.imag(Xhat[i,1]), np.real(Xhat[i,1]),
                edgecolor = plt.cm.Set1(clus[i][0]), facecolor = plt.cm.Set1(clus[i][0]),
                s = 5, zorder = 1)

ax.set_title('Data for autoencoder')
ax.set_xlabel('$Re(\\hat{u}_2)$')
ax.set_ylabel('$Im(\\hat{u}_1)$')
ax.set_zlabel('$Re(\\hat{u}_1)$')
ax.axes.set_xlim3d(left=-150, right=150) 
ax.axes.set_ylim3d(bottom=-150, top=150) 
ax.axes.set_zlim3d(bottom=-150, top=150)

# Plot the dynamics data
Xdynhat = np.fft.fft(Xdyn)
fig = plt.figure()
ax = p3.Axes3D(fig)

# Plot randomly sampled points from dynamics dataset
fs = 2 # plot every fs points
for i in range(nClus):
    Xtemp = Xdynhat[labelDyn == i, :]
    Xtemp = Xtemp[np.random.randint(0, len(Xtemp), int(len(Xtemp)/fs)), :]
    ax.scatter(np.real(Xtemp[:, 2]), np.imag(Xtemp[:, 1]), np.real(Xtemp[:, 1]),
               facecolor = plt.cm.Set1(i),
                s = 5, edgecolor = plt.cm.Set1(i), zorder = 1)

# Plot all points resulting from overlap
for i in range(nDyn):
    for j in range(len(clusDyn[i]) - 1, 0, -1):
        ax.scatter(np.real(Xdynhat[i,2]), np.imag(Xdynhat[i,1]), np.real(Xdynhat[i,1]),
                edgecolor = plt.cm.Set1(clusDyn[i][j]), facecolor = (0, 0, 0, 0),
                s = j*j*40, linewidth = 2, zorder = -j)
        ax.scatter(np.real(Xdynhat[i,2]), np.imag(Xdynhat[i,1]), np.real(Xdynhat[i,1]),
                edgecolor = plt.cm.Set1(clusDyn[i][0]), facecolor = plt.cm.Set1(clusDyn[i][0]),
                s = 5, zorder = 1)
        
ax.set_title('Data for dynamics')
ax.set_xlabel('$Re(\\hat{u}_2)$')
ax.set_ylabel('$Im(\\hat{u}_1)$')
ax.set_zlabel('$Re(\\hat{u}_1)$')
ax.axes.set_xlim3d(left=-150, right=150) 
ax.axes.set_ylim3d(bottom=-150, top=150) 
ax.axes.set_zlim3d(bottom=-150, top=150)     

# Plot original data, its reconstruction, and difference between the two
x = np.append(np.squeeze(mat['x']), 2*np.pi)
t -= t[0]
fig, ax = plt.subplots(1, 3)
c = ax[0].contourf(t, x, np.append(X, X[:, 0:1], axis = 1).transpose(), levels = np.linspace(-6, 6, 13), cmap = 'RdBu_r')
ax[0].set_title('Data')
fig.colorbar(c, ax = ax[0], ticks = [-6, 0, 6])
Xrecon = np.zeros(X.shape)
for i in range(nSamples):
    Xrecon[i, :] = chartMaps[clus[i][0]].decode(chartMaps[clus[i][0]].encode(X[i:i + 1,:]))
c = ax[1].contourf(t, x, np.append(Xrecon, Xrecon[:, 0:1], axis = 1).transpose(), levels = np.linspace(-6, 6, 13), cmap = 'RdBu_r')
ax[1].set_title('Decode(Encode(Data))')
fig.colorbar(c, ax = ax[1], ticks = [-6, 0, 6])
c = ax[2].contourf(t, x, np.append(X - Xrecon, (X - Xrecon)[:, 0:1], axis = 1).transpose(), cmap = 'RdBu_r')
ax[2].set_title('Error')
fig.colorbar(c, ax = ax[2])

ax[0].set_xlabel('t')
ax[0].set_ylabel('x')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x')
ax[2].set_xlabel('t')
ax[2].set_ylabel('x')

# Dynamics part
# Evolve an initial condition forward in time using the charts and dynamics 
# on them
x0 = X[0:1, :]

# Find which cluster the point is in initially, map into local coordinates
clusNew = kmeans.predict(x0)[0]
y = chartMaps[clusNew].encode(x0)

# Evolve the point forward in time
nsteps = 6564
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
t = np.arange(0, (nsteps + 1)*0.05, 0.05)
fig, ax = plt.subplots()
c = ax.contourf(t, x, np.append(xArr, xArr[:, 0:1], axis = 1).transpose(), levels = np.linspace(-6, 6, 13), cmap = 'RdBu_r')
ax.set_title('Evolved trajectory')
fig.colorbar(c, ax = ax, ticks = [-6, 0, 6])
ax.set_xlabel('t')
ax.set_ylabel('x')

# Plot trajectory in Fourier space
xArrhat = np.fft.fft(xArr)
clusArr = np.squeeze(clusArr)
dif = np.squeeze(np.nonzero(np.diff(clusArr)))
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot(np.real(xArrhat[0:dif[0] + 1, 2]), np.imag(xArrhat[0:dif[0] + 1, 1]), np.real(xArrhat[0:dif[0] + 1, 1]),
        color = plt.cm.Set1(clusArr[0]), linewidth = 1)
for i in range(len(dif) - 1):
    ax.plot(np.real(xArrhat[dif[i]:dif[i + 1] + 1, 2]), np.imag(xArrhat[dif[i]:dif[i + 1] + 1, 1]), np.real(xArrhat[dif[i]:dif[i + 1] + 1, 1]),
        color = plt.cm.Set1(clusArr[dif[i] + 1]), linewidth = 1)
ax.plot(np.real(xArrhat[dif[-1]:nsteps + 1, 2]), np.imag(xArrhat[dif[-1]:nsteps + 1, 1]), np.real(xArrhat[dif[-1]:nsteps + 1, 1]),
        color = plt.cm.Set1(clusArr[dif[-1] + 1]), linewidth = 1)
ax.set_title('Evolved trajectory')
ax.set_xlabel('$Re(\\hat{u}_2)$')
ax.set_ylabel('$Im(\\hat{u}_1)$')
ax.set_zlabel('$Re(\\hat{u}_1)$')
ax.axes.set_xlim3d(left=-150, right=150) 
ax.axes.set_ylim3d(bottom=-150, top=150) 
ax.axes.set_zlim3d(bottom=-150, top=150)
