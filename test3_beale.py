import my_bfgs
import numpy as np

"""
    Test 3 :    Beale's Function (n = 5)
                Global Minimum is [0, 0]

                f(x)  = Beale's Function
                f'(x) = Derivative of Beale's Function (Self-Calculated)
"""


def f(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


def fp(x):
    return np.array([(2*(1.5 - x[0] + x[0]*x[1])*(-1 + x[1]) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(-1 + x[1]**2)
                      + 2*(2.625 - x[0] + x[0]*x[1]**3)*(-1 + x[1]**3)), (2*(1.5 - x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0]
                      + x[0]*x[1]**2)*(2*x[1]*x[0]) + 2*(2.625 - x[0] + x[0]*x[1]**3) * (3*x[1]**2*x[0]))], dtype=float)

n = 2
root = np.array([3, 0.5]).reshape(2, 1)

x0 = np.array([2, 2]).reshape(2, 1)

tol = 1e-6
max_its = 1000

my_bfgs.bfgs(x0, f, fp, tol, max_its, root)