"""
Limited functionality Akima-spline-based numerical analysis methods.
Package is shared-memory only. User must wrap this package with an MPI-parallel
data analysis class for distributed memory computing.
Assumes 1-D domain decomposition.

Notes:
------

Definitions:
------------

Authors:
--------
Colin Towery

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""
import numpy as np
from numba import jit
from scipy.interpolate import Akima1DInterpolator as _interp

__all__ = ['flux_diff', 'interp3d', 'interp1d', '_deriv']


def _deriv(phi, h, axis=0):
    """
    deriv(phi, h, axis=0):

    deriv computes the k'th derivative of a uniform gridded array along the
    prescribed axis using Akima spline approximation.

    Arguments
    ---------
    phi   - input array
    h     - uniform grid spacing
    axis  -

    Output
    ------
    f - d^k/dx^k(phi)
    """
    if phi.ndim != 3:
        print("ERROR: phi.ndim not equal to 3!")

    axis = axis % phi.ndim
    if axis != 2:
        phi = np.swapaxes(phi, axis, 2)

    s = list(phi.shape)
    x = np.arange(-3, s[2]+3, dtype=phi.dtype)
    xh = np.arange(-0.5, s[2]+0.5, 1.0, dtype=phi.dtype)
    deriv = np.empty_like(phi)

    nx = s[2] + 6
    tmp = np.empty(nx, dtype=phi.dtype)

    for k in range(s[0]):
        for j in range(s[1]):
            tmp[3:-3] = phi[k, j, :]
            tmp[:3] = phi[k, j, -3:]
            tmp[-3:] = phi[k, j, :3]

            spline = _interp(x, tmp)
            phih = spline(xh)
            deriv[k, j, :] = (1.0/h)*(phih[1:] - phih[:-1])

    if axis != 2:
        deriv = np.swapaxes(deriv, axis, 2)

    return deriv


@jit(nopython=True, nogil=True)
def flux_diff(var, dx, axis=0, ng=3):
    """
    1st order difference of interpolated midpoints along given axis of var.
    """
    assert var.ndim == 3, "ERROR: var.ndim not equal to 3!"
    assert ng > 2, "ERROR: a minimum 3 ghost cells on each side are required!"

    dx_inv = 1.0/dx

    shape = list(var.shape)
    shape[axis] -= 2*ng
    nx, ny, nz = shape
    out = np.empty((nx, ny, nz), dtype=var.dtype)

    axis = axis % var.ndim
    axes = ((axis+1) % 3, (axis+2) % 3, axis)

    varT = var.transpose(axes)   # new _view_ into the inputs
    outT = out.transpose(axes)   # new _view_ into the outputs

    # if outT isn't a view, return is "empty"
    # assert np.may_share_memory(out, outT)

    nx0, nx1, nx2 = varT.shape
    x = np.arange(nx2).astype(var.dtype)

    # midpoints surrounding internal data points
    xh = 0.5*(x[ng-1:-ng] + x[ng:-ng+1])

    # since axis 2 of varT may not actually be contiguous, use temp array
    for k in range(nx0):
        for j in range(nx1):
            temp = np.ascontiguousarray(varT[k, j])
            varh = interp1d(x, temp, xh)
            outT[k, j] = dx_inv*(varh[1:] - varh[:-1])

    return out


@jit(nopython=True, nogil=True, cache=True)
def interp3d(x0, x1, x2, f, xp):
    """
    Returns scalar function value at a single point in three-dimensional
    space using tri-cubic Akima spline interpolation
    """
    # x0 = np.asarray(x0)
    # x1 = np.asarray(x1)
    # x2 = np.asarray(x2)
    # f = np.asarray(f)
    # xp = np.asarray(xp)
    nk, nj, ni = f.shape

    # assert x0.ndim == x1.ndim == x2.ndim == 1  # , 'x0, x1, x2 must be 1D!'
    # assert xp.size == 3  # , 'xp must have size 3!'

    # assert x0.size == nk
    # assert x1.size == nj
    # assert x2.size == ni
    # assert nk == nj and nj == ni and ni == 6

    tk = np.empty(nk, dtype=f.dtype)
    tj = np.empty(nj, dtype=f.dtype)

    for k in range(nk):
        for j in range(nj):
            tj[j] = interp1d(x2, f[k, j], np.array([xp[2]]))[0]
        tk[k] = interp1d(x1, tj, np.array([xp[1]]))[0]

    fi = interp1d(x0, tk, np.array([xp[0]]))

    return fi


@jit(nopython=True, nogil=True, cache=True)
def interp1d(x, f, xi):
    """
    Returns scalar function values at any number of points in
    one-dimensional space using cubic Akima spline interpolation
    """
    # x = np.asarray(x)
    # f = np.asarray(f)
    # xi = np.array([xi]).ravel()

    # assert x.size >= 6
    # assert np.all(xi >= x[2]) and np.all(xi <= x[-3])

    dx = x[1:] - x[:-1]
    m = (f[1:] - f[:-1]) / dx    # m.size = x.size - 1
    dm = np.abs(m[1:] - m[:-1])  # dm.size = x.size - 2
    a = dm[2:]   # a = |m[3]-m[2]|, |m[4]-m[3]|
    b = dm[:-2]  # b = |m[1]-m[0]|, |m[2]-m[1]|
    ab = a + b

    # t.size = x.size - 4, also scipy uses ab > 1e-9*ab.max() for some reason
    t = np.where(ab > 0.0, (a*m[1:-2] + b*m[2:-1]) / ab,
                 0.5*(m[1:-2] + m[2:-1]))

    # p1.size = p2.size = p3.size = x.size - 5
    p1 = t[:-1]
    p2 = (3.0*m[2:-2] - 2.0*t[:-1] - t[1:]) / dx[2:-2]
    p3 = (t[:-1] + t[1:] - 2.0*m[2:-2]) / dx[2:-2]**2

    idx = np.searchsorted(x, xi, side='right') - 1
    w = xi - x[idx]
    p0 = f[idx]
    fmin = np.fmin(f[idx], f[idx+1])
    fmax = np.fmax(f[idx], f[idx+1])

    idx -= 2
    # polynomial version:
    fi = p0 + p1[idx]*w + p2[idx]*w**2 + p3[idx]*w**3

    # fused multiply-add version:
    # fi = p0 + (p1[idx] + (p2[idx] + p3[idx]*w)*w)*w

    # preserve extrema -- not a part of the original Akima spline!!
    fi = np.fmin(fmax, np.fmax(fi, fmin))

    return fi
