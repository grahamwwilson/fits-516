from scipy import special
import numpy as np

#
# pendulumT, pendulumT2, and pendulumT3 are all basically the same 
# model and describe the HighPrecision data
#
# pendulumT2 is probably the most appropriate way of 
# writing the problem.
#
# pendulumT and pendulumT3 are closer to how I originally 
# formulated the model for the purpose of simulating 
# the data-sets.
#

def pendulumT(x, T0, theta0, beta):
#
# Period of finite amplitude simple pendulum with fixed factor 
# of decrease in amplitude per complete cycle (beta). 
# The parameters are (T0, theta0, beta).
#
# T0 [s]               : Period for infinitesimal amplitude
# theta0 [rad]         : Initial angle to the vertical
# beta [dimensionless] : Amplitude decrease factor per complete cycle
#
    theta = theta0*pow(beta,x)    # Decrease angle by a factor of beta after each complete cycle
    k = np.sin(0.5*theta)         # k parameter - related to angular amplitude at the start of each cycle
# Calculate corresponding finite amplitude period
    period = T0*(2.0/np.pi)*special.ellipk(k*k) # Use formula (2) from p3 of Pendulum Experiment writeup

    return period 

def pendulumT2(x, T0, theta0, tau):
#
# Period of finite amplitude simple pendulum with exponential decrease 
# in amplitude per complete cycle (tau). 
# The parameters are (T0, theta0, tau)
#
# T0 [s]               : Period for infinitesimal amplitude
# theta0 [rad]         : Initial angle to the vertical
# tau [cycles]         : Amplitude exponential time constant
#
    theta = theta0*np.exp(-x/tau)   # Decrease angle exponentially with time constant of tau cycles
    k = np.sin(0.5*theta)           # k parameter - related to angular amplitude at the start of each cycle
# Calculate corresponding finite amplitude period
    period = T0*(2.0/np.pi)*special.ellipk(k*k) # Use formula (2) from p3 of Pendulum Experiment writeup

    return period 

def pendulumT3(x, g, L, theta0deg, beta):
#
# Period of finite amplitude simple pendulum with fixed factor of 
# decrease in amplitude per complete cycle (beta). 
# The parameters are (g, L, theta0, beta). 
# It is not feasible to determine gravity and L independently from this data.
# So you need to fix either g or L to determine the other (T0 =2pi/sqrt(g/L))
# Written this way so that you can see what uncertainty on g would be implied 
# by this data if the pendulum length were known exactly.
#
# g [m/s^2]            : Gravitational acceleration
# L [m]                : Simple pendulum length
# theta0deg [degrees]  : Initial angle to the vertical (in degrees)
# beta [dimensionless] : Amplitude decrease factor per complete cycle
#
    theta0 = theta0deg*np.pi/180.0
    theta = theta0*pow(beta,x)      # Decrease angle by a factor of beta after each complete cycle
    k = np.sin(0.5*theta)           # k parameter - related to angular amplitude at the start of each cycle
# Calculate corresponding finite amplitude period
    w0sq = g/L
    T0 = 2.0*np.pi/np.sqrt(w0sq)
    period = T0*(2.0/np.pi)*special.ellipk(k*k) # Use formula (2) from p3 of Pendulum Experiment writeup

    return period 
