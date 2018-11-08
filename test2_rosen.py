import my_bfgs
import numpy as np
from scipy.optimize import rosen, rosen_der
import matplotlib.pyplot as plt

"""
Test 2 :    Rosenbrock Function (n = 3)
            Global Minimum is [1, 1, 1]

            f(x)  = rosen(x)
            f'(x) = rosen_der(x)
"""


def f(x):
    return rosen(x)


def fp(x):
    return rosen_der(x).reshape(3, 1)

n = 3
root = np.array([1, 1, 1]).reshape(3, 1)

x0 = np.array([2, 2, 2]).reshape(3, 1)

tol = 1e-6
max_its = 1000

my_bfgs.bfgs(x0, f, fp, tol, max_its, root)

my_bfgs.bfgs(x0, f, fp, tol, max_its, root, plot=True, plt_range=[-3, 3, -3, 3])