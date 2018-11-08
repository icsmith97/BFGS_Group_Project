import my_bfgs
import numpy as np

"""
    Test 4 :    Booth Function (n = 2)
                Global Minimum is [1, 3]

                f(x)  = (x[0] + 2*x[1] - 7)^2 + (2*x[0] + x[1] -5)^2
                f'(x) = [2(x[0] + 2*x[1] - 7), 2(2*x[0] + x[1] - 5)]
"""


def f(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def fp(x):
    return np.array([10*x[0] + 8*x[1] - 34, (8*x[0] + 10*x[1] - 38)], dtype=float).reshape(2, 1)


n = 2
root = np.array([1, 3]).reshape(2, 1)

x0 = np.array([20, 43]).reshape(2, 1)

tol = 1e-6
max_its = 1000

my_bfgs.bfgs(x0, f, fp, tol, max_its, root)