#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from sklearn.kernel_ridge import KernelRidge
import keras
import tensorflow as tf

class chartMap:
    """Map and its inverse for a chart"""
    
    def __init__(self, nDim):
        self.nDim = nDim
    
    def train(self, X):
        pass
    
    def encode(self, X):
        pass
    
    def decode(self, X):
        pass
    

class svdkrr(chartMap):
    """PCA for encoder, kernel ridge regression for decoder"""
    
    # Train the encoder and decoder. Shape of X is (nSamples, nFeatures).
    def train(self, X, **kwargs):
        # Compute SVD for decoder
        self.X0 = np.mean(X, axis = 0)
        U, _, _ = linalg.svd((X - self.X0).transpose(), full_matrices = False)
        self.U = U[:, 0:self.nDim]

        # Perform kernel ridge regression to reconstruct points in full space
        self.clf = KernelRidge(**kwargs)
        self.clf.fit(self.encode(X), X - self.X0)
        
    # Map X to lower dimension. Shape of X is (nSamples, nFeatures). Shape of 
    # output is (nSamples, nDim).
    def encode(self, X):
        return np.matmul(self.U.transpose(), (X - self.X0).transpose()).transpose()
        
    # Inverse map back to ambient dimension. Shape of Y is (nSamples, nDim).
    # Shape of output is (nSamples, nFeatures). 
    def decode(self, Y):
        return self.clf.predict(Y) + self.X0
    
    
class autoencoder(chartMap):
    """Neural network for encoder, neural network for decoder"""
    
    # Build the encoder and decoder
    def build(self, n, enStruct, enAct, deStruct, deAct):
        """
        Build the encoder and decoder

        Parameters
        ----------
        n : integer
            Dimension of state vector
        enStruct : list of integers
            Dimensions of hidden layers in encoder
        enAct : list of strings, same length as enStruct
            Activation functions in encoder
        deStruct : list of integers
            Dimensions of hidden and output layers in decoder
        deAct : list of strings, same length as deStruct
            Activation functions in decoder

        Returns
        -------
        None.

        """
        
        # Check inputs
        if len(enStruct) != len(enAct) or len(deStruct) != len(deAct) or deStruct[-1] != n:
            raise ValueError('Incorrect autoencoder structure')
        
        # Set precision
        keras.backend.set_floatx('float64')
        
        # Assemble encoder
        enInput = keras.Input(shape = (n,))
        enState = enInput
        for i in range(len(enStruct)):
            enState = keras.layers.Dense(enStruct[i], activation = enAct[i])(enState)
        
        # Assemble decoder
        deState = enState
        for i in range(len(deStruct)):
            deState = keras.layers.Dense(deStruct[i], activation = deAct[i])(deState)
            
        # Build autoencoder as well as separate encoder and decoder
        self.autoEnc = keras.Model(inputs = enInput, outputs = deState)
        self.encoder = keras.Model(enInput, enState)
        deInput = keras.Input(shape = (enStruct[-1],))
        deState = deInput
        for layer in self.autoEnc.layers[-len(deStruct):]:
            deState = layer(deState)
        self.decoder = keras.Model(deInput, deState)
        
    # Train the autoencoder
    def train(self, X, optArgs = {}, trainArgs = {}):
        self.autoEnc.compile(**optArgs)
        self.history = self.autoEnc.fit(X, X, **trainArgs)
        
    # Map X to lower dimension. Shape of X is (nSamples, nFeatures). Shape of 
    # output is (nSamples, nDim).
    def encode(self, X):
        return self.encoder.predict(X)
        
    # Inverse map back to ambient dimension. Shape of Y is (nSamples, nDim).
    # Shape of output is (nSamples, nFeatures). 
    def decode(self, Y):
        return self.decoder.predict(Y)


class hybridnn(chartMap):
    """Hybrid neural network from Linot & Graham (PRE 2020)"""
    
    # Get PCA change-of-basis matrix. Shape of X is (nSamples, nFeatures).
    def trainPCA(self, X):
        self.X0 = np.mean(X, axis = 0)
        self.U, _, _ = linalg.svd((X - self.X0).transpose(), full_matrices = True)
    
    # Build the encoder and decoder
    def build(self, n, enStruct, enAct, deStruct, deAct, alpha):
        """
        Build the encoder and decoder

        Parameters
        ----------
        n : integer
            Dimension of state vector
        enStruct : list of integers
            Dimensions of hidden layers in encoder
        enAct : list of strings, same length as enStruct
            Activation functions in encoder
        deStruct : list of integers
            Dimensions of hidden and output layers in decoder
        deAct : list of strings, same length as deStruct
            Activation functions in decoder
        alpha : double
            tradeoff in loss (see eq. 4 in Linot & Graham)

        Returns
        -------
        None.

        """
        
        # Check inputs
        if len(enStruct) != len(enAct) or len(deStruct) != len(deAct) or deStruct[-1] != n:
            raise ValueError('Incorrect autoencoder structure')
        
        # Set precision
        keras.backend.set_floatx('float64')
        
        # Get PCA change-of-basis matrix and mean
        U = keras.backend.variable(value = self.U)
        X0 = keras.backend.variable(value = self.X0)
        
        # Encoder
        enInput = keras.Input(shape = (n,))
        enState = enInput
        
        # Change to PCA coordinates
        enState = keras.layers.Lambda(lambda x: x - X0, name = 'SubMean')(enState)
        pcaState = keras.layers.Lambda(lambda x: tf.einsum("ij,jk->ik", x, U), name = 'ChangeBasisIn')(enState)
        
        # Assemble encoder
        pcaStateTrunc = keras.layers.Lambda(lambda x: tf.slice(x, [0, 0], [-1, self.nDim]), name = 'Truncate')(pcaState)
        enState = pcaState
        for i in range(len(enStruct)):
            enState = keras.layers.Dense(enStruct[i], activation = enAct[i], name = 'DenseIn' + str(i))(enState)
        
        enStateInt = enState # save this intermediate state for later use
            
        # Add results from PCA coordinates and encoder
        enState = keras.layers.Add(name = 'AddIn')([enState, pcaStateTrunc])
        
        # Decoder
        # Pad encoded state with zeros
        paddings = tf.constant([[0, 0,], [0, n - self.nDim]])
        pcaState = keras.layers.Lambda(lambda x: tf.pad(x, paddings, 'CONSTANT'), name = 'AppendZeros')(enState)
    
        # Assemble decoder
        deState = enState
        for i in range(len(deStruct)):
            deState = keras.layers.Dense(deStruct[i], activation = deAct[i], name = 'DenseOut' + str(i))(deState)
            
        # Compute state for second part of loss function from eq. 4 in Linot & Graham)
        deStateTrunc = keras.layers.Lambda(lambda x: tf.slice(x, [0, 0], [-1, self.nDim]), name = 'Extract')(deState)
        zeroState = keras.layers.Add()([deStateTrunc, enStateInt])

        # Add results from PCA coordinates and decoder
        deState = keras.layers.Add(name = 'AddOut')([deState, pcaState])
        
        # Change basis and add mean back in
        deState = keras.layers.Lambda(lambda x: tf.einsum("ij,kj->ik", x, U), name = 'ChangeBasisOut')(deState)
        deState = keras.layers.Lambda(lambda x: x + X0, name = 'AddMean')(deState)
            
        # Build autoencoder and add loss function
        self.autoEnc = keras.Model(inputs = enInput, outputs = deState)
        self.autoEnc.add_loss(alpha*tf.reduce_mean(tf.square(zeroState)))
        
        # Build separate encoder and decoder
        self.encoder = keras.Model(enInput, enState)
        deInput = keras.Input(shape = (self.nDim,))
        deState = deInput
        for i in range(len(deStruct)):
            deState = self.autoEnc.get_layer('DenseOut' + str(i))(deState)
        pcaState = self.autoEnc.get_layer('AppendZeros')(deInput)
        deState = self.autoEnc.get_layer('AddOut')([deState, pcaState])
        deState = self.autoEnc.get_layer('ChangeBasisOut')(deState)
        deState = self.autoEnc.get_layer('AddMean')(deState)
        self.decoder = keras.Model(deInput, deState)
        
    # Train the autoencoder
    def train(self, X, optArgs = {}, trainArgs = {}):
        self.autoEnc.compile(**optArgs)
        self.history = self.autoEnc.fit(X, X, **trainArgs)
        
    # Map X to lower dimension. Shape of X is (nSamples, nFeatures). Shape of 
    # output is (nSamples, nDim).
    def encode(self, X):
        return self.encoder.predict(X)
        
    # Inverse map back to ambient dimension. Shape of Y is (nSamples, nDim).
    # Shape of output is (nSamples, nFeatures). 
    def decode(self, Y):
        return self.decoder.predict(Y)