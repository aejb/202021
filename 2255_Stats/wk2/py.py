import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib
import matplotlib.pyplot as plt

h   = np.array([1000., 828.,  800.,  600.,  300.])
d   = np.array([1500., 1340., 1328., 1172., 800.])
sig = np.array([15.,   15.,   15.,   15.,   15.])

# First hypothesis
def f1(h, *vars):
    a, = vars
    return a*h

## Curve fit
p0_1 = np.array([1.0])
theta_hat_1, covariance_1 = curve_fit(f1, h, d, p0_1, sig, absolute_sigma=True)

## Standard deviation of parameters
theta_hat_1_std_dev = np.sqrt(np.diag(covariance_1))


# Second hypothesis
def f2(h, *vars):
    a, b = vars
    return a*h + b*h**2

## Curve fit
p0_2 = np.array([1.0, 1.0])
theta_hat_2, covariance_2 = curve_fit(f2, h, d, p0_2, sig, absolute_sigma=True)

## Standard deviation of parameters
theta_hat_2_std_dev = np.sqrt(np.diag(covariance_2))

# Third hypothesis
def f3(h, *vars):
    a, b = vars
    return a*h**b

## Curve fit
p0_3 = np.array([1.0, 1.0])
theta_hat_3, covariance_3 = curve_fit(f3, h, d, p0_3, sig, absolute_sigma=True)

## Standard deviation of parameters
theta_hat_3_std_dev = np.sqrt(np.diag(covariance_3))

# Plot graphs

## Setup Plots
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
ax1 = fig1.add_subplot()
ax1.set_title('first hypothesis ($\\alpha h$) fit')
ax2 = fig2.add_subplot()
ax2.set_title('second hypothesis ($\\alpha h+\\beta h^2$) fit')
ax3 = fig3.add_subplot()
ax3.set_title('third hypothesis ($\\alpha h^\\beta$)fit')
ax1.set_ylabel("d")
ax2.set_ylabel("d")
ax3.set_ylabel("d")
ax1.set_xlabel("h")
ax2.set_xlabel("h")
ax3.set_xlabel("h")

h_val = np.arange(0, 1000, 1)

def std_dev(h, cov): return np.sqrt(sum([(h**(i+j))*cov[i][j] for i in range(len(cov)) for j in range(len(cov))]))

## Given Data
ax1.errorbar(h, d, yerr=sig, fmt='kx', label="test data")
ax2.errorbar(h, d, yerr=sig, fmt='kx', label="test data")
ax3.errorbar(h, d, yerr=sig, fmt='kx', label="test data")

## Plot the three fits
fit1 = np.vectorize(lambda h, a: a*h)
ax1.plot(h_val, fit1(h_val, theta_hat_1), 'r', label="first hypothesis fit")
ax1.fill_between(h_val, fit1(h_val, theta_hat_1)-std_dev(h_val, covariance_1), fit1(h_val, theta_hat_1)+std_dev(h_val, covariance_1), label="one standard deviation")
ax1.legend()

fit2 = np.vectorize(lambda h, a, b: a*h + b*h**2)
ax2.plot(h_val, fit2(h_val, *theta_hat_2), 'c', label="second hypothesis fit")
ax2.fill_between(h_val, fit2(h_val, *theta_hat_2)-std_dev(h_val, covariance_2), fit2(h_val, *theta_hat_2)+std_dev(h_val, covariance_2), label="one standard deviation")
ax2.legend()

fit3 = np.vectorize(lambda h, a, b: a*h**b)
ax3.plot(h_val, fit3(h_val, *theta_hat_3), 'y', label="third hypothesis fit")
ax3.fill_between(h_val, fit3(h_val, *theta_hat_3)-std_dev(h_val, covariance_3), fit3(h_val, *theta_hat_3)+std_dev(h_val, covariance_3), label="one standard deviation")
ax3.legend()

# Chi-Square

## Hypothesis 1
chi_sq_min_1 = sum(((d - f1(h, *theta_hat_1))/sig)**2)
ndof_1 = len(h)-len(p0_1)
print(f"---Hypothesis 1---\nChi-sq-min/NDoF:\t{chi_sq_min_1/ndof_1}\np-value:\t\t{chi2.sf(chi_sq_min_1, df=ndof_1)}")

## Hypothesis 2
chi_sq_min_2 = sum(((d - f2(h, *theta_hat_2))/sig)**2)
ndof_2 = len(h)-len(p0_2)
print(f"---Hypothesis 2---\nChi-sq-min/NDoF:\t{chi_sq_min_2/ndof_2}\np-value:\t\t{chi2.sf(chi_sq_min_2, df=ndof_2)}")

## Hypothesis 3
chi_sq_min_3 = sum(((d - f3(h, *theta_hat_3))/sig)**2)
ndof_3 = len(h)-len(p0_3)
print(f"---Hypothesis 3---\nChi-sq-min/NDoF:\t{chi_sq_min_3/ndof_3}\np-value:\t\t{chi2.sf(chi_sq_min_3, df=ndof_3)}")