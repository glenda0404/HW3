import numpy as np
import sympy as sp
import math
from scipy.interpolate import lagrange

def lagrange_interpolation(x_values, y_values, x_target):
    polynomial = lagrange(x_values, y_values)
    return polynomial(x_target)

def error_bound(x_values, x_target):
    """
    Computes an upper bound for the interpolation error using the formula:
    E_n(x) = max|f^(n+1)(ξ)| * Π |(x - x_i)| / (n+1)!
    """
    n = len(x_values) - 1
    factorial_term = math.factorial(n+1)
    product_term = np.prod([abs(x_target - x) for x in x_values])

    # The (n+1)th derivative of cos(x) is sin(x) or -sin(x)
    max_derivative = 1  # Since |sin(x)| <= 1 always

    error = (max_derivative * product_term) / factorial_term
    return error

# Given data points
x_vals = np.array([0.698, 0.733, 0.768, 0.803])
y_vals = np.array([0.7661, 0.7432, 0.7193, 0.6946])
x_target = 0.750

# Compute interpolations for different degrees
for degree in range(1, 5):
    interp_x_vals = x_vals[:degree+1]
    interp_y_vals = y_vals[:degree+1]
    approx_value = lagrange_interpolation(interp_x_vals, interp_y_vals, x_target)
    err_bound = error_bound(interp_x_vals, x_target)
    print(f"Degree {degree}: Approximation = {approx_value:.6f}, Error Bound = {err_bound:.6e}")
