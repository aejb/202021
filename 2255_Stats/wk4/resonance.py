"""
Resonance.py

Illustrate resonance dataset from a CSV file and fit the resonance peak.

@author: Lev Levitin <l.v.levitin@rhul.ac.uk>
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load raw data: column 1 = f, column 2 = V1, column 3 = V2, column 4 = phase
# Load the data. The unpack=True argument transposes the 2D data array,
# allowing to initialise the data columns as separate variables.
f, V1, V2, phase = np.genfromtxt(r'resonance.csv', unpack=True, usecols=(0,1,2,3), delimiter=',')
omega = 2*np.pi*f
Vratio2 = (V2/V1)**2

# circuit parameter, fixed in the fit
R = 100

# initial guess for estimators
Rprime = 50
Q = 2
omega0 = 10e6

# fit model
def RLC_model(omega, omega0, Q, Rprime):
    return R**2 / (R + Rprime)**2 / (1 + Q**2 * (omega/omega0 - omega0/omega)**2)

p0 = [omega0, Q, Rprime]

# fit the model using Levenberg-Marquardt algorithm
popt, pcov = curve_fit(RLC_model, omega, Vratio2, p0=p0)
# popt is the array of fit parameters
omega0, Q, Rprime = popt

# Plot the data and annotate the graph
plt.title(r'''Frequency Responce of RLC Circuit''')
ax1 = plt.subplot(2,1,1)
ax1.set_xlabel(r'Frequency $\omega/2\pi$ (kHz)')
ax1.set_ylabel(r'$(V_2 / V_1)^2$')

# chose a scaling factor for frequencies: convert omega in rad/sec into f in kHz:
f_scaling = 1e-3 / (2*np.pi)

ax1.scatter(omega * f_scaling, Vratio2, marker='s', color='blue', label='data')

omegaPlot = np.linspace(min(omega), max(omega), 1000)
ax1.plot(omegaPlot * f_scaling, RLC_model(omegaPlot, omega0, Q, Rprime), color='r',
         label=r"LCR model: $f_0 = %.0f$kHz, $Q=%.1f$, $R'=%.1f\Omega$" % (omega0 * f_scaling, Q, Rprime))

ax1.legend()

ax2 = plt.subplot(2,1,2)
ax2.set_xlabel(r'Frequency $\omega/2\pi$ (kHz)')
ax2.set_ylabel(r'Phase ($^\circ$)')
ax2.scatter(omega * f_scaling, phase, marker='s', color='blue', label='data')

plt.tight_layout()
plt.show()

