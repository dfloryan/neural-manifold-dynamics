# CANDyMan
Data-driven discovery of intrinsic dynamics

README for CANDyMan:
Charts and Atlases for Nonlinear Data-Driven Dynamics on Manifolds

This distribution contains code that implements an atlas of charts in the 
context of data-driven forecasting of dynamical systems, as described in
"Data-driven discovery of intrinsic dynamics," by D. Floryan and M. D. Graham, 
arXiv:2108.05928, 2021. 

# Contents
This distribution contains two primary Python modules: 
* chartMap.py: implements an atlas of charts
* overlap.py: makes charts overlap

This distribution also contains the data needed to recreate the results in 
the cited paper, and seven Python scripts that recreate the main results 
and demonstrate how to implement and use the method:
* exampleCircle.py: recreates main results for circle data
* exampleTorus1d.py: recreates main results for periodic torus data
* exampleTorus2d.py: recreates main results for quasiperiodic torus data
* exampleKSbeating.py: recreates main results for beating Kuramoto-Sivashinsky data
* exampleKSbeatingTravelling.py: recreates main results for beating travelling Kuramoto-Sivashinsky data
* exampleKSbursting.py: recreates main results for bursting Kuramoto-Sivashinsky data
* exampleReactDifNeumann.py: recreates main results for reaction-diffusion data

Note: the reaction-diffusion dataset is too large to store on Github. 
In its place, Matlab code that will produce the dataset has been provided. 
Once created, place the dataset in the same directory as 
exampleReactDifNeumann.py. 

# Dependencies and system
All code was written in Python 3. The following libraries are used:
* NumPy (1.21.5)
* SciPy (1.7.3)
* Keras (2.3.1)
* TensorFlow (2.0.0)
* scikit-learn (1.0.2)
* Matplotlib (3.5.1)

The code was tested on a MacBook Pro, Apple M1 chip, macOS Monterey 12.4.

# Installation and execution
Simply download the contents of this repository and run an example script.
Execution times for the examples vary from O(seconds) to O(10 minutes), 
depending on the example. Labelled output plots will be created upon 
execution of an example script. 

# Reference
If you make use of this distribution, please cite "Data-driven discovery of 
intrinsic dynamics," by D. Floryan and M. D. Graham, arXiv:2108.05928, 2021. 
