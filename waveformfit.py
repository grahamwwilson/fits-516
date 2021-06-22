#
# Adapt standard example to fitting leading edge 
# or trailing edge of waveform data from TDS220 oscilloscope. 
#
# The input data file is a time series 
# with 2444 voltage measurements in time consisting of:
# [Column 0] time-ordered measurement number 
# [Column 1] ADC value [0-255] (8-bits)
# [Column 2] time (seconds)
# [Column 3] Voltage (V)
# corresponding to a captured trigger of the oscilloscope.
#
# Data was acquired with a Si PIN photodiode observing a laser-beam 
# that is chopped with an optical chopper running at 45 Hz.
# The expected shape is that of the Gaussian cumulative distribution 
# function whose standard deviation (in time) is related to the laser beam size 
# and the relative speed of the laser spot with respect to the chopper.
#
# Use fitchoice=100 for rising edge and fitchoice=101 for falling edge.
#
#                   Graham W. Wilson,   22-JUN-2021
# 
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from scipy import stats
from iminuit import Minuit
from iminuit.util import describe, make_func_code
from pprint import pprint

import mymodels as mf         # Simple model functions
import pendulummodels as pmf  # Pendulum model functions
from MyLeastSquares import LeastSquares, LsqDriver # Put all the chi-squared calculations here

# 1. First setup the data
# Read data from file into numpy array format

#filewithlocalpath='WaveFormFile_Rising.dat'
#filewithlocalpath='WaveFormFile_Falling.dat'
filewithlocalpath='WaveFormFile_Falling_100us.dat'

x_data, y_data = genfromtxt(filewithlocalpath, usecols=(2,3),unpack=True)
x_data = x_data*1.0e6  # Convert to micro-seconds
errorval = 0.05*1.151  # Hard-code what you want here or do your own error evaluation. 
print('Setting errors on y-values manually to ',errorval)
y_err = np.full(x_data.shape, errorval)

# 2. Secondly specify the fit
# Need to specify the model, reasonable starting values, and whether parameters are fixed.

#fitchoice=100  # rising
fitchoice=101  # falling

if fitchoice==100:
   lsq = LsqDriver(mf.srcurve, x_data, y_data, y_err)  # for "rising S-curve"
else:
   lsq = LsqDriver(mf.sfcurve, x_data, y_data, y_err)  # for "falling S-curve"

m = Minuit(lsq, x0=250.0, sigx=100.0, yamplitude=8.0, yoffset=0.0, fix_yoffset=False, pedantic=True, print_level=2, errordef=1.0)

# 3. Add some plotting customization
SMALL_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 32
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# 4. First plot the data with assigned errors
plt.figure(1)    
errorbar(x_data,y_data,y_err,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
title('TDS220 Waveform')
xlabel('Time [us]')
ylabel('Voltage [V]')
plt.show()

# 5. Do the fit
m.migrad(ncall=100000)

# 6. Check how many parameters are fitted (ie not fixed) in the fit
nfitted = 0
nfixed = 0
for p in m.parameters:
    if m.fixed[p] == True:
       print('Parameter ',p,' is fixed ')
       nfixed +=1
    else:
       print('Parameter ',p,' is fitted ')
       nfitted +=1

# 7. Do error analysis
if nfitted>0:  # Before proceeding with hesse and error analysis, make sure that there are free parameters to fit
   m.hesse(ncall=100000)

# 8. Hypothesis test results based solely on chi-squared
print(' ')
print('A) Hypothesis test: IS THIS A REASONABLE FIT? ')
print('chi-squared                          : ', m.fval)
print('number of data points                : ', len(y_data))
print('total number of fit parameters       : ', nfitted+nfixed, 
      ' of which ',nfitted,' are fitted and ',nfixed,'are fixed')
print('number of ACTUAL fitted parameters   : ', nfitted)
ndof = len(y_data)- nfitted
print('number of degrees of freedom (d.o.f.): ', ndof)

pvaluepercent = 100.0*(1.0 - stats.chi2.cdf(m.fval, ndof ))
print(' ')
print('Observed chi-squared p-value of',pvaluepercent,'%')

# 9. Parameter Estimation results including covariance matrix and correlations
print(' ')
print('B) Parameter Estimation: Given a reasonable fit, what are the model parameters and uncertainties?')
print('Fitted parameters:')
for p in m.parameters:
    if m.fixed[p] == False:
       print("{} = {} +- {} ".format(p,m.values[p], m.errors[p] ))
    else:
       print("{} = {}  (Fixed = {})".format(p,m.values[p], m.fixed[p] ))
print(' ')

if nfitted > 0:  # Likewise only do this if there are free parameters
   print('Covariance matrix')
   pprint(m.matrix())
   print(' ')
   print('Correlation Coefficient Matrix')
   print(m.matrix(correlation=True))
   print(' ')

# 10. Use fitted parameter values to evaluate Run Test statistic
rpval = lsq.runspvalue(*m.values.values())  # this is a method in MyLeastSquares.py
print('Observed run test p-value (%) = ',rpval)

# 11. Combined test assuming that the two tests (chi-squared and run test) 
# are independent and using the continuous approximation. See Eadie 11.6 for more details.
p1 = 0.01*pvaluepercent
p2 = 0.01*rpval
pcomb = 0.0
if p1*p2>0.0:
   pcomb = p1*p2*(1.0-np.log(p1*p2))
print('Combined p-value of chi-squared and run-test (%) = ',100.0*pcomb,'(uses continuous approximation)')

# 12. Model values and fit deviations
if fitchoice==100:   # make sure model that was fitted is superimposed
   y_model = mf.srcurve(x_data, *m.values.values())
elif fitchoice==101:
   y_model = mf.sfcurve(x_data, *m.values.values())

y_dev = y_data - y_model    # deviations (data - fit-model)

# 13. Plots using the best fit parameters
plt.figure(2)     #Data and Fit
errorbar(x_data,y_data,y_err,fmt="o",color='blue',solid_capstyle='projecting',capsize=2)
plt.plot(x_data, y_model, color='red')
title('TDS220 Waveform')
xlabel('Time [us]')
ylabel('Voltage [V]')

plt.figure(3)     #Residuals
errorbar(x_data,y_dev,y_err,fmt="o",color='blue',solid_capstyle='projecting',capsize=2)
plt.plot(x_data, mf.linemodel(x_data, 0.0, 0.0), color='red')
title('TDS220 Waveform')
xlabel('Time [us]')
ylabel('Voltage Residual [V]')

plt.figure(4)     #Normalized Residuals (with errors of 1.0 sigma) (y_data - y_model)/y_err
errorbar(x_data,y_dev/y_err,1.0,fmt="o",color='blue',solid_capstyle='projecting',capsize=2)
plt.plot(x_data, mf.linemodel(x_data, 0.0, 0.0), color='red')
title('TDS220 Waveform')
xlabel('Time [us]')
ylabel('Fit Normalized Residual (s.d.)')

plt.show()
