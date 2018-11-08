import numpy as np                      # Contains functions for linear algebra and numerical computation.
import time                             # Contains functions for timing code.
import matplotlib.pyplot as plt         # Contains functions to plot graphs.
from matplotlib import rc               # Allows access to the LaTeX interpreter for formatting graphs.

"""
 Summary: This file implements the BFGS algorithm, will plot a log error graph and if the function is two dimensional
          will produce a contour plot showing the iterates.
"""

# Defines font settings for the LaTeX interpreter used for plotting graphs.
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def backtracking_line_search(f, grad_fk, pk, xk):
    a = 1
    rho = 0.5
    c = 0.1

    # Starting with a = 1, then chose values for rho and c in (0,1).

    while f(xk + (a * pk)) > (f(xk) + (c * a) * (grad_fk.transpose() @ pk)):
        a = rho * a

    return a


def bfgs_update(Hk, rhok, sk, yk, n):
    rsy = (rhok * sk) @ np.transpose(yk)
    rys = (rhok * yk) @ np.transpose(sk)
    rss = (rhok * sk) @ np.transpose(sk)

    hkp1 = (np.eye(n) - rsy) @ Hk @ (np.eye(n) - rys) + rss

    return hkp1


def bfgs(x0, f, fprime, tol, max_its, root, **kwargs):
    """
    :param x0: starting value for x_k
    :param f: the function to be minimized
    :param fprime: gradient of the function to be minimized
    :param tol: gradient norm must be less than tol before successful termination
    :param max_its: maximum number of iterations before termination
    :param root: err_plot needs a root to work

    Optional Parameters

    :param plot: boolean that determines whether a contour plot is produced
    :param plt_range: range of the contour plot's x and y axes
    :param levels: number of contour levels to plot

    :var n : integer
    :var pk: column vector
    :var grad_fk: column vector
    :var xk: column vector
    :var sk: column vector

    :return x_n: the approximated root
    :return time: time to find root
    :return its: the number of iterations performed
    """

    plot = kwargs.get('plot', False)
    plt_range = kwargs.get('plt_range', [-5, 5, -5, 5])
    contour_levels = kwargs.get('levels', 500)

    n = len(x0)     # use the starting value to determine the dimension of the problem

    xk = x0

    Hk = np.eye(n)

    grad_fk = fprime(xk)

    grad_norm = np.linalg.norm(grad_fk)

    st = time.time()

    k = 0

    points_so_far = [xk]

    while (grad_norm > tol):
        pk = - Hk @ grad_fk

        ak = backtracking_line_search(f, grad_fk, pk, xk)

        xkp1 = xk + ak * pk

        sk = xkp1 - xk

        grad_fkp1 = fprime(xkp1)

        yk = grad_fkp1 - grad_fk

        # Nocedal 8.20 suggests rescaling h0 before performing the first BFGS update

        if k == 0:
            Hk = np.inner(yk, sk) / np.inner(yk, yk) * np.eye(n)

        # BFGS update is performed
        try:
            rhok = 1.0 / (np.dot(yk.transpose(), sk))

            if rhok > 1e10:
                raise ValueError
        except ZeroDivisionError:
            rhok = 1000.0
        except ValueError:
            rhok = 1000.0

        Hk = bfgs_update(Hk, rhok, sk, yk, n)

        # prepare for the next iteration

        xk = xkp1
        points_so_far.append(xk)
        grad_fk = grad_fkp1
        grad_norm = np.linalg.norm(grad_fk)
        k += 1

    x_n = xk

    time_taken = time.time() - st

    its = k

    # We return the approximated root, the time taken to find it, and the number of
    # iterations taken to find it

    errs = []

    for pt in points_so_far:
        er = np.linalg.norm(pt - root)
        errs.append(er)

    log_error1 = np.log(errs[0: len(errs) - 1])
    log_error2 = np.log(errs[1: len(errs)])

    plot_x = np.array(log_error1)
    plot_y = np.array(log_error2)

    err_grad = np.polyfit(plot_x, plot_y, 1)[0]

    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$\log e_k$', fontsize=14)
    plt.ylabel(r'$\log e_{k+1}$', fontsize=14)
    plt.grid(True)
    plt.title(r'Log Error Plot', fontsize=16)
    plt.plot(plot_x, plot_y, 'ro')

    if plot:
        x = np.linspace(plt_range[0], plt_range[1], 500)
        y = np.linspace(plt_range[2], plt_range[3], 500)
        x, y = np.meshgrid(x, y)

        z = f(np.array([x, y]))

        x_pts = []
        y_pts = []

        for pt in points_so_far:
            x_pts.append(pt[0])
            y_pts.append(pt[1])

        plt.figure(2)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$x_1$', fontsize=14)
        plt.ylabel(r'$x_2$', fontsize=14)
        plt.title(r'Contour Plot of $f(x)$', fontsize=16)
        plt.contour(x, y, z, contour_levels, linewidths=0.5)
        plt.grid(True)
        plt.plot(x_pts, y_pts, 'xb-')


    print("Approximated Root: ", x_n.transpose())
    print("Time Taken: ", time_taken)
    print("Iterations: ", its)
    print("Gradient of Error Plot: ", err_grad)

    plt.show()

    input()

    return x_n, time_taken, its, err_grad

# Just testing GitHub
