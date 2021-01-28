# mymodels.py
# Various simple model functions
# Add your own ...

import numpy as np

def linemodel(x, slope, intercept):  # simple straight line model with explicit parameters
    return slope*x + intercept

def quadmodel(x, a0, a1, a2):        # quadratic model with explicit parameters
    return a0 + a1*x + a2*x**2

def exponential(x, c, tau):          # exponential model
    return c*np.exp(-x/tau) 
