import numpy as np
import numpy.polynomial.hermite as Hermite
from scipy import integrate
from scipy.misc import derivative

# Nth order wave function
n = 1
# Use natural units
hbar = 1
m = 1
w = 1
# Define "a" substitution
a = np.sqrt(hbar / (m*w))

# Define Hermite polynomial generator
def hermite(x, n):
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Hermite.hermval(x/a, herm_coeffs)
  
# Define wave function (split into prefactor for readability)
def psi(x, n):
    prefactor = 1./np.sqrt(2.**n * np.math.factorial(n) * np.sqrt(np.pi*a**2))
    return prefactor * np.exp(- x**2 / (2*a**2)) * hermite(x,n)

## Code to generate expectation values (Section 4.1)
# Iterate over quantum states 0 to 10
for order in range(0, 11):
    # Define callable for each quantum state
    def psi_n(x): return psi(x, order)
    # Position expectation values
    ex_x = integrate.quad(np.vectorize(lambda x: psi(x, order)*x*psi(x, order)), -float("inf"), float("inf"))[0]
    ex_x2 = integrate.quad(np.vectorize(lambda x: psi(x, order)*x**2*psi(x, order)), -float("inf"), float("inf"))[0]
    # Momentum expectation values
    ex_p = -hbar*1j*integrate.quad(np.vectorize(lambda x : psi_n(x) * derivative(psi_n, x, dx=0.0001, n=1)), -float("inf"), float("inf"))[0]
    ex_p2 = -hbar**2*integrate.quad(np.vectorize(lambda x : psi_n(x) * derivative(psi_n, x, dx=0.0001, n=2)), -float("inf"), float("inf"))[0]
    print(f"------\torder: {order}\t------")
    # Uncertanties
    Dx = np.math.sqrt(ex_x2 - ex_x**2)
    Dp = np.math.sqrt(ex_p2 - abs(ex_p)**2)
    DxDp = Dx*Dp
    print(f"Dx\t{Dx}\nDp\t{Dp}")
    print(f"DxDp\t{DxDp}")