# CANDyMan
Charts and Atlases for Nonlinear Data-Driven Dynamics on Manifolds

README for Charts and Atlases for Nonlinear Data-Driven Dynamics on Manifolds

This distribution contains code that implements an atlas of charts in the 
context of data-driven forecasting of dynamical systems, as described in
"Charts and atlases for nonlinear data-driven models of dynamics on manifolds," 
by D. Floryan and M. D. Graham, arXiv:2108.05928, 2021. 

This distribution contains two primary Python functions: 
* chartMap.py: implements an atlas of charts
* overlap.py: makes charts overlap

This distribution also contains the data needed to recreate the results in 
the cited paper, and six Python scripts that recreate the main results 
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

All code was written in Python 3. The following libraries are used:
* NumPy
* SciPy
* Keras
* TensorFlow
* scikit-learn
* collections
* Matplotlib
* mpl_toolkits

If you make use of this distribution, please cite "Charts and atlases for 
nonlinear data-driven models of dynamics on manifolds," by D. Floryan and 
M. D. Graham, arXiv:2108.05928, 2021. 
