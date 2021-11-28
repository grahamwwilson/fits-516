# fits-516
Example code for parameter estimation and goodness of fit tests for PHSX 516
The code is python3 (not python2).

Need numpy, scipy, iminuit, matplotlib etc installed.
Initial versions (old laptop) and my new laptop configuration (can check using pip3 list or pip list)
numpy: 1.15.4/1.17.4
scipy: 1.1.0/1.3.3
iminuit: 1.5.4/2.8.4
matplotlib: 3.0.2/3.1.2

The iminuit 1->2 transition has some backwards incompatible changes to the API.

To execute
python3 example.py

The code uses the supplied model files: mymodels.py, pendulummodels.py 
and the fitting class LeastSquares (in MyLeastSquares.py).
Provided is an example data set, PeriodHighPrecision.dat.

example.out is the output you should obtain
