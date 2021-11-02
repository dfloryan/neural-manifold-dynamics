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

# Load beating travelling wave data
nSamples = 100
mat = scipy.io.loadmat('ksdataBeatingTravelling.mat')
X = mat['udata'].transpose()

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
# Phase-align the data so that first spatial Fourier mode is purely real
Xhat = np.fft.fft(X)
phi = np.angle(Xhat[:, 1])
wav = np.concatenate((np.arange(33), np.arange(-31, 0))) # wavenumbers
XhatShift = Xhat*np.exp(-1j*np.outer(phi, wav))
Xshift = np.real(np.fft.ifft(XhatShift))
print("Compute k-means clustering...")
nClus = 3
kmeans = KMeans(n_clusters = nClus, random_state = 0).fit(Xshift)
label = kmeans.labels_
print("Done")

# Compute how phase changes between snapshots. This will be used in the 
# dynamics portion of the model. 
dphi = phi[1:] - phi[:-1]
dphi += (dphi < -np.pi)*2.0*np.pi - (dphi > np.pi)*2.0*np.pi # remove jumps

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
nDim = 1 # dimension to reduce to (for the shape function)

print("Compute maps for each cluster...")
chartMaps = np.empty((nClus, ), dtype = object)
chartDyn = np.empty((nClus, ), dtype = object)
phaseDyn = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    chartMaps[i] = autoencoder(nDim)
    enStruct = [128, 64, 16, 8, nDim]
    enAct = ['elu', 'elu', 'elu', 'elu', None]
    deStruct = [8, 16, 64, 128, Xshift.shape[1]]
    deAct = ['elu', 'elu', 'elu', 'elu', None]
    chartMaps[i].build(Xshift.shape[1], enStruct, enAct, deStruct, deAct)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 2000, 'batch_size': Xshift[clusPts[i], :].shape[0]}
    chartMaps[i].train(Xshift[clusPts[i], :], optArgs, trainArgs)
    
    # Find the shape dynamics
    nnStruct = [32, 32, 16, 4, nDim]
    nnAct = ['elu', 'elu', 'elu', 'elu', None]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 2000, 'batch_size': Xshift[clusPts[i], :].shape[0] - 1}
    ind = clusPts[i][np.nonzero(np.diff(clusPts[i]) == 1)] # indices for first snapshots in snapshot pairs
    chartDyn[i] = nnDyn(nDim, nnStruct, nnAct, chartMaps[i].encode(Xshift[ind, :]), chartMaps[i].encode(Xshift[ind + 1, :]), optArgs, trainArgs)
    
    # Find the phase dynamics
    nnStruct = [32, 32, 16, 4, 1]
    nnAct = ['elu', 'elu', 'elu', 'elu', None]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 200, decay_rate = 0.9, staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    optArgs = {'optimizer': optimizer, 'loss': 'mean_squared_error'}
    trainArgs = {'epochs': 2000, 'batch_size': Xshift[clusPts[i], :].shape[0] - 1}
    phaseDyn[i] = nnDyn(nDim, nnStruct, nnAct, chartMaps[i].encode(Xshift[ind, :]), dphi[ind], optArgs, trainArgs)

print("Done")

# Store encoded versions of all points
XshiftEncode = np.empty((nClus, ), dtype = object)
for i in range(nClus):
    XshiftEncode[i] = chartMaps[i].encode(Xshift[clusPts[i], :])
    
# Plot original data, its reconstruction, and difference between the two
x = np.append(np.squeeze(mat['x']), 2*np.pi)
t = np.squeeze(mat['t'])
fig, ax = plt.subplots(1, 3)
c = ax[0].contourf(t, x, np.append(Xshift, Xshift[:, 0:1], axis = 1).transpose(), levels = np.linspace(-12, 12, 23), cmap = 'RdBu_r')
ax[0].set_title('Data')
fig.colorbar(c, ax = ax[0], ticks = [-12, 0, 12])
XshiftRecon = np.zeros(Xshift.shape)
for i in range(nSamples):
    XshiftRecon[i, :] = chartMaps[clus[i][0]].decode(chartMaps[clus[i][0]].encode(Xshift[i:i + 1,:]))
c = ax[1].contourf(t, x, np.append(XshiftRecon, XshiftRecon[:, 0:1], axis = 1).transpose(), levels = np.linspace(-12, 12, 23), cmap = 'RdBu_r')
ax[1].set_title('Decode(Encode(Data))')
fig.colorbar(c, ax = ax[1], ticks = [-12, 0, 12])
c = ax[2].contourf(t, x, np.append(Xshift - XshiftRecon, (Xshift - XshiftRecon)[:, 0:1], axis = 1).transpose(), cmap = 'RdBu_r')
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
x0 = Xshift[0:1, :]
phi0 = phi[0:1]

# Find which cluster the point is in initially, map into local coordinates
clusNew = kmeans.predict(x0)[0]
y = chartMaps[clusNew].encode(x0)

# Evolve the point forward in time
nsteps = 10000
yArr = np.zeros((nsteps + 1, nDim))
yArr[0, :] = y
xArr = np.zeros((nsteps + 1, Xshift.shape[1]))
xArr[0, :] = x0
phiArr = np.zeros((nsteps + 1, 1))
phiArr[0, :] = phi0
clusArr = np.zeros((nsteps + 1, 1), dtype = int)
clusArr[0, 0] = clusNew
for i in range(nsteps):
    # Map points forward
    phiArr[i + 1, 0] = phiArr[i, 0] + phaseDyn[clusNew].predict(y)
    y = chartDyn[clusNew].predict(y)
    
    # Find nearest training point in chart. Switch charts if necessary.
    clusOld = clusNew
    dist2 = np.sum((XshiftEncode[clusNew] - y)**2, axis = 1)
    clusNew = clus[clusPts[clusNew][np.argmin(dist2)]][0]
    if clusNew != clusOld:
        y = chartMaps[clusNew].encode(chartMaps[clusOld].decode(y))
    
    # Store trajectory
    yArr[i + 1, :] = y
    xArr[i + 1, :] = chartMaps[clusNew].decode(y)
    clusArr[i + 1, 0] = clusNew

# Add phase to shape, plot the trajectory
xArrHat = np.fft.fft(xArr)
xArrHat = xArrHat*np.exp(1j*np.outer(phiArr, wav))
xArrShift = np.real(np.fft.ifft(xArrHat))

fig, ax = plt.subplots()
t = np.arange(0, (nsteps + 1)*0.01, 0.01)
c = ax.contourf(t, x, np.append(xArrShift, xArrShift[:, 0:1], axis = 1).transpose(), levels = np.linspace(-12, 12, 23), cmap = 'RdBu_r')
ax.set_title('Evolved trajectory')
fig.colorbar(c, ax = ax, ticks = [-12, 0, 12])
ax.set_xlabel('t')
ax.set_ylabel('x')
    