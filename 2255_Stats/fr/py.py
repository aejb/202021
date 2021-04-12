import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

## Defines constants in cm
# Laser wavelength
_w = 6.238e-5
# Single-Slit-Screen distance and error
_l = 166
_dl = 1
# Observational error
_dx = 0.05
# Double-Slit seperation
_a = 0.0250

## Defines single-slit and double-slit functions
# Single-slit function
def ss_func(n, *vars):
    b, = vars
    return _l*( (_w/b)*n)
# Double-slit function
def ds_func(n, *vars):
    l, = vars
    return l*( (_w/_a)*(n+1/2))

## Define datasets 
# n: order of intensity maxima/minima
# for maxima, +1/2 must be added to the n-value of the minima
n = np.array([])
# m: horizontal x-offset of the intensity maxima/minima
m = np.array([])
# s: error of the measurement
s = np.array([])

## Curve-fitting functions
# Single-slit
# Define starting value for unknown variable
p0 = np.array([1e-4])
# Calculate parameter estimate and convariance
parameters, covariance = curve_fit(ss_func, n, m, p0, s, absolute_sigma=True)
# Calculate chi-squared-min
chi_sq_min = sum(((m-ss_func(n, parameters))/s)**2)

#Double-slit
p0 = np.array([1])
parameters, covariance = curve_fit(ds_func, n, m, p0, s, absolute_sigma=True)
chi_sq_min = sum(((m-ds_func(n, parameters))/s)**2)