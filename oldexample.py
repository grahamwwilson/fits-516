#
# More complete fitting examples for PHSX516.
#
# 1. Separate choice of data and choice of fit.
# 2. Illustrate how to set errors by hand (see L37-44)
# 3. Include plotting in this example too
#
# The basic example reads in the 100 data points from the 
# simulated data file PeriodHighPrecision.dat and 
# does a chi-squared fit using the uncertainties from the file.
# Models 8,9,10,11 and 12 are all reasonable models to test.
#
# Note the syntax for fixing parameters.
# In addition to the normal chi-squared fit, the deviations are also 
# analyzed using the Run Test.
#
#                   Graham W. Wilson,   28-JAN-2021
#
# Update 28-NOV-2021. This is the version that works with iminuit 1.5.4.
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

# Check numpy version
print("numpy version",np.__version__)

datachoice = 2    # Set this to the data file of interest
fitchoice = 11    # Set this to whichever one you want to test
# Good practice to print what was selected (and save the output file)
print('datachoice = ',datachoice)
print('fitchoice  = ',fitchoice)

# 1. First setup the data
if datachoice == 1:
# Use data from our first example (straight line fit)
   x_data = np.array([0.000, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.000])
   y_data = np.array([0.993, 1.208, 1.526, 1.615, 1.962, 2.168, 2.307, 2.555, 2.863, 2.967])
   y_err =  np.array([0.050, 0.045, 0.040, 0.045, 0.035, 0.045, 0.045, 0.050, 0.040, 0.050])
else:
# Read data from file into numpy array format
   filewithlocalpath='PeriodHighPrecision.dat'
   x_data, y_data = genfromtxt(filewithlocalpath, usecols=(0,1),unpack=True)
   errorsfromfile = 1 # Normal setting is 1. To hard-code the error value set not equal to 1.  
   if errorsfromfile == 1:
      print('Reading errors on y-values from file')  # Standard operation 
      y_err = genfromtxt(filewithlocalpath, usecols=(2),unpack=True)
   else:   
      errorval = 0.314159  # hard-code what you want here or do your own error propagation. 
      print('Setting errors on y-values manually to ',errorval)
      y_err = np.full(x_data.shape, errorval)

# 2. Secondly specify the fit
# Need to specify the model, reasonable starting values, and whether parameters are fixed.
if fitchoice<=4:
   lsq = LsqDriver(mf.linemodel, x_data, y_data, y_err)
   if fitchoice==1:
      m = Minuit(lsq, slope=2.0, intercept=1.0, pedantic=True, print_level=2, errordef=1.0)
   elif fitchoice==2:
      m = Minuit(lsq, slope=2.0, intercept=1.0, fix_slope=True, pedantic=True, print_level=2, errordef=1.0)
   elif fitchoice==3:
      m = Minuit(lsq, slope=2.0, intercept=1.0, fix_intercept=True, pedantic=True, print_level=2, errordef=1.0)
   elif fitchoice==4:
      m = Minuit(lsq, slope=2.0, intercept=1.0, fix_slope=True, fix_intercept=True, pedantic=True, print_level=2, errordef=1.0)
elif fitchoice==7:
   lsq = LsqDriver(mf.quadmodel, x_data, y_data, y_err)
   m = Minuit(lsq, a0=1.0, a1=2.0, a2=0.0, pedantic=True, print_level=2, errordef=1.0)
elif fitchoice==8:
   lsq = LsqDriver(pmf.pendulumT, x_data, y_data, y_err)
   m = Minuit(lsq, T0=2.5, theta0=1.0, beta=0.98, pedantic=True, print_level=2, errordef=1.0)
elif fitchoice==9:
   lsq = LsqDriver(pmf.pendulumT2, x_data, y_data, y_err)
   m = Minuit(lsq, T0=2.5, theta0=1.0, tau=50.0, pedantic=True, print_level=2, errordef=1.0)
elif fitchoice<=12:
# This is with the correct model, but with 3 parameters fitted to the data
   lsq = LsqDriver(pmf.pendulumT3, x_data, y_data, y_err)
   if fitchoice==10: # fit (g, theta0, beta)
      m = Minuit(lsq, g=9.8, L=1.5, theta0deg=50.0, beta=0.98, fix_L=True, pedantic=True, print_level=2, errordef=1.0)
   elif fitchoice==11: # fit (L, theta0, beta)
      m = Minuit(lsq, g=9.81, L=1.4, theta0deg=50.0, beta=0.98, fix_g=True, pedantic=True, print_level=2, errordef=1.0)
   elif fitchoice==12:
# This is the truth model (all parameters fixed)
      m = Minuit(lsq, g=9.81, L=1.5, theta0deg=60.0, beta=0.98, fix_g=True, fix_L=True, fix_theta0deg=True, 
                 fix_beta=True, pedantic=True, print_level=2, errordef=1.0)

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
title('My Plot Title')
xlabel('My Plot x-label')
ylabel('My plot y-label')
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
if fitchoice <=4 :    # make sure model that was fitted is superimposed
   y_model = mf.linemodel(x_data, *m.values.values())
elif fitchoice==7 :
   y_model = mf.quadmodel(x_data, *m.values.values())
elif fitchoice==8 :
   y_model = pmf.pendulumT(x_data, *m.values.values())
elif fitchoice==9 :
   y_model = pmf.pendulumT2(x_data, *m.values.values())
elif fitchoice>=10 and fitchoice<=12 :
   y_model = pmf.pendulumT3(x_data, *m.values.values())

y_dev = y_data - y_model    # deviations (data - fit-model)

# 13. Plots using the best fit parameters
plt.figure(2)     #Data and Fit
errorbar(x_data,y_data,y_err,fmt="o",color='blue',solid_capstyle='projecting',capsize=5)
plt.plot(x_data, y_model, color='red')
title('Pendulum Period Data (HP)')
xlabel('Oscillation Number')
ylabel('Period(s)')

plt.figure(3)     #Residuals
errorbar(x_data,y_dev,y_err,fmt="o",color='blue',solid_capstyle='projecting',capsize=5)
plt.plot(x_data, mf.linemodel(x_data, 0.0, 0.0), color='red')
title('Pendulum Period Data (HP)')
xlabel('Oscillation Number')
ylabel('Period Residual(s)')

plt.figure(4)     #Normalized Residuals (with errors of 1.0 sigma) (y_data - y_model)/y_err
errorbar(x_data,y_dev/y_err,1.0,fmt="o",color='blue',solid_capstyle='projecting',capsize=5)
plt.plot(x_data, mf.linemodel(x_data, 0.0, 0.0), color='red')
title('Pendulum Period Data (HP)')
xlabel('Oscillation Number')
ylabel('Period Fit Normalized Residual (sigmas)')

plt.show()
