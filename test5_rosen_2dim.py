import my_bfgs
import numpy as np
from scipy.optimize import rosen, rosen_der
import matplotlib.pyplot as plt

"""
Test 2 :    Rosenbrock Function (n = 2)
            Global Minimum is [1, 1]

            f(x)  = rosen(x)
            f'(x) = rosen_der(x)
"""


def f(x):
    return rosen(x)


def fp(x):
    return rosen_der(x).reshape(2, 1)

n = 2
root = np.array([1, 1]).reshape(2, 1)

x0 = np.array([-2, 3]).reshape(2, 1)

tol = 1e-6
max_its = 1000

my_bfgs.bfgs(x0, f, fp, tol, max_its, root, plot=True, plt_range=[-5, 5, -5, 5])

my_bfgs.bfgs(x0, f, fp, tol, max_its, root)