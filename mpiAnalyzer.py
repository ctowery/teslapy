"""
Description:
============
This module contains the mpiAnalyzer object classes for the TESLaCU package.
It should not be imported unless "__main__" has been executed with MPI.

Notes:
======

Indexing convention:
--------------------
Since TESLa has mostly worked in MATLAB and Fortran, it is common for us to
think in terms of column-major index order, i.e., [x1, x2, x3], where x1 is
contiguous in memory and x3 is always the inhomogenous dimension in the
Athena-RFX flame geometry.
However, Python and C/C++ are natively row-major index order, i.e.
[x3, x2, x1], where x1 remains contiguous in memory and x3 remains the
inhomogenous dimension.

The TESLaCU package adheres to row-major order for indexing data grids and
when indexing variables associated with the grid dimensions (e.g. nx, ixs,
etc.), however all vector fields retain standard Einstein notation indexing
order, (e.g. u[0] == u1, u[1] == u2, u[2] == u3).

Coding Style Guide:
-------------------
This module generally adheres to the Python style guide published in
PEP 8, with the following exceptions:
- Warning W503 (line break occurred before a binary operator) is
  ignored, since this warning is a mistake and PEP 8 recommends breaking
  before operators
- Error E225 (missing whitespace around operator) is ignored

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Additionally, I have not yet, but plan to eventually get all of these
docstrings whipped into shape and in compliance with the Numpy/Scipy
style guide for documentation:
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>

Finally, this module should always strive to achieve enlightenment by
following the Zen of Python (PEP 20, just `import this` in a Python
shell) and using idiomatic Python (i.e. 'Pythonic') concepts and design
patterns.

Authors:
========
Colin Towery

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""

from mpi4py import MPI
import numpy as np
# from math import sqrt
import os
import sys
# from memory_profiler import profile

from . import fft as tcfft          # FFT transforms and math functions
from . import stats as tcstats      # statistical functions
# from .diff import central as tcfd   # finite difference functions
from .diff import akima as tcas     # Akima spline approximation functions

__all__ = []


# -----------------------------------------------------------------------------
def mpiAnalyzer(comm=MPI.COMM_WORLD, odir='./analysis/', pid='test',
                ndims=3, N=[512]*3, L=[2*np.pi]*3,
                decomp=None, periodic=None, config=None,
                method='akima_flux_diff', **kwargs):
    """[summary]

    The mpiAnalyzer() function is a "class factory" which returns the
    appropriate mpi-parallel analyzer class instance based upon the
    inputs. Each subclass specializes the BaseAnalyzer for a different
    problem configuration that can be added to the if/elif branch of this
    factory.

    Parameters
    ----------
    **kwargs :
        additional arguments documented by the subclasses
    comm : {MPI.Comm}, optional
        MPI communicator for the analyzer (the default is MPI.COMM_WORLD)
    odir : {str}, optional
        [description] (the default is './analysis/', which
        [default_description])
    pid : {str}, optional
        [description] (the default is 'test', which [default_description])
    ndims : {number}, optional
        [description] (the default is 3, which [default_description])
    N : {list}, optional
        [description] (the default is [512]*3, which [default_description])
    L : {list}, optional
        [description] (the default is [2*np.pi]*3, which [default_description])
    decomp : {[type]}, optional
        [description] (the default is None, which [default_description])
    periodic : {[type]}, optional
        [description] (the default is None, which [default_description])
    config : {str}, optional
        [description] (the default is 'hit', which [default_description])
    method : {str}, optional
        [description] (the default is 'akima_flux_diff', which
        [default_description])

    Returns
    -------
    _baseAnalyzer object
        Single instance of _baseAnalyzer or one of its subclasses
    """
    """
    odir: output directory of analysis products
    pid: problem ID (file prefix)
    ndims: number of spatial dimensions of global data (not subdomain)
    L: scalar or tuple of domain dimensions
    N: scalar or tuple of mesh dimensions
    config: problem configuration (switch)
    kwargs:

    Output:
    -------

    """

    if decomp in [2, 3] and method != 'akima_flux_diff':
        print("WARNING: for 2D and 3D domain decompositions only Akima-spline"
              " flux differencing is currently available.")
        method = 'akima_flux_diff'

    if config == 'hit':
        if decomp not in [None, 1]:
            print("WARNING: HIT configuration requires 1D domain "
                  "decomposition in order to use the spectral analysis tools")

        if periodic not in [None, True, ndims*(True, ), ndims*[True, ]]:
            print("WARNING: HIT configuration assumes fully-periodic "
                  "boundary conditions. User-supplied periodicity condition "
                  "will be ignored.")

        analyzer = _hitAnalyzer(comm, odir, pid, ndims, decomp, L, N, method)

    elif config is None:
        analyzer = _baseAnalyzer(comm, odir, pid, ndims, decomp, periodic,
                                 L, N, method)
    else:
        if comm.rank == 0:
            print("mpiAnalyzer.factory configuration arguments not recognized!"
                  "\nDefaulting to base analysis class: _baseAnalyzer().")
        analyzer = _baseAnalyzer(comm, odir, pid, ndims, L, N)

    return analyzer


# -----------------------------------------------------------------------------
class _baseAnalyzer(object):
    """
    class _baseAnalyzer(object) ...
    Empty docstring!
    """

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, comm, odir, pid, ndims, decomp, periodic, L, N, method):

        # --------------------------------------------------------------
        # Unprocessed user-input instance properties
        self._pid = pid
        self._ndims = ndims
        self._config = "Unknown (Base Configuration)"
        self._odir = odir
        init_comm = comm

        # --------------------------------------------------------------
        # Make analysis output directory
        if init_comm.rank == 0:
            try:
                os.makedirs(odir)
            except OSError as e:
                if not os.path.isdir(odir):
                    raise e
                else:
                    status = e
            finally:
                if os.path.isdir(odir):
                    status = 0
        else:
            status = None

        status = init_comm.bcast(status)
        if status != 0:
            MPI.Finalize()
            sys.exit(999)

        # --------------------------------------------------------------
        # Processed user-input instance properties
        if np.iterable(N):
            if len(N) == 1:
                self._nx = np.array(list(N)*ndims, dtype=np.int)
            elif len(N) == ndims:
                self._nx = np.array(N, dtype=np.int)
            else:
                raise ValueError("The length of N must be either 1 or ndims")
        elif np.isscalar(N):
            self._nx = np.array([N]*ndims, dtype=np.int)
        else:  # I think None and dicts might be what's left
            raise ValueError("Unrecognized argument type! N should be a "
                             "scalar or array_like.")

        if np.iterable(L):
            if len(L) == 1:
                self._L = np.array(list(L)*ndims, dtype=np.float64)
            elif len(L) == ndims:
                self._L = np.array(L, dtype=np.float64)
            else:
                raise ValueError("The length of L must be either 1 or ndims")
        elif np.isscalar(L):
            self._L = np.array([L]*ndims, dtype=np.float64)
        else:  # I think None and dicts might be what's left
            raise ValueError("Unrecognized argument type! L should be a "
                             "scalar or array_like.")

        if decomp is None:
            self._decomp = 1
        elif decomp in [1, 2, 3] and decomp <= ndims:
            self._decomp = decomp
        else:
            raise ValueError("decomp must be 1, 2, 3 or None")

        if periodic is None:
            self._periodic = tuple([False]*ndims)
        elif np.iterable(periodic):
            if len(periodic) == ndims:
                self._periodic = tuple(periodic)
            elif len(periodic) == 1:
                self._periodic = ndims*tuple(periodic)
            else:
                raise ValueError("The length of periodic must be either 1 "
                                 "or ndims")
        elif np.isscalar(periodic):
            self._periodic = ndims*(periodic, )
        else:
            raise ValueError("Unrecognized argument type! Periodic should "
                             "be None, a scalar, or array_like")

        if method == 'central_diff':
            self.deriv = self._centdiff_deriv
        elif method == 'akima_flux_diff':
            self.deriv = self._akima_slab_deriv
        elif method == 'ignore':
            self.deriv = None
        else:
            if init_comm.rank == 0:
                print("mpiAnalyzer._baseAnalyzer.__init__(): "
                      "'method' argument not recognized!\n"
                      "Defaulting to Akima spline flux differencing.")
            self.deriv = self._akima_slab_deriv

        self._dx = self._L/self._nx

        # --------------------------------------------------------------
        # MPI domain decomposition
        dims = MPI.Compute_dims(init_comm.size, self._decomp)
        dims.extend([1]*(ndims-self._decomp))

        assert np.all(np.mod(self._nx, dims) == 0)

        self._comm = init_comm.Create_cart(dims, self._periodic)
        self._nnx = self._nx//dims
        self._ixs = self._nnx*self.comm.coords
        self._ixe = self._ixs+self._nnx

        # --------------------------------------------------------------
        # other stuff (DO NOT count on these being permanent!)
        self.tol = 1.0e-6
        self.prefix = pid
        self.moments_file = '%s/%s.moments' % (self.odir, self.prefix)

    # -------------------------------------------------------------------------
    # Class Properities
    # -------------------------------------------------------------------------
    def __enter__(self):
        # with statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with statement finalization
        return False

    @property
    def comm(self):
        return self._comm

    @property
    def odir(self):
        return self._odir

    @property
    def pid(self):
        return self._pid

    @property
    def ndims(self):
        return self._ndims

    @property
    def config(self):
        return self._config

    @property
    def periodic(self):
        return self._periodic

    @property
    def L(self):
        return self._L

    @property
    def nx(self):
        return self._nx

    @property
    def dx(self):
        return self._dx

    @property
    def nnx(self):
        return self._nnx

    @property
    def ixs(self):
        return self._ixs

    @property
    def ixe(self):
        return self._ixe

    # -------------------------------------------------------------------------
    # Statistical Moments
    # -------------------------------------------------------------------------
    def psum(self, data):
        return tcstats.psum(data)

    def central_moments(self, data, w=None, wbar=None, m1=None):
        """
        Computes global min, max, and 1st to 6th biased central moments of
        assigned spatial field. To get raw moments, simply pass in m1=0.
        """
        return tcstats.central_moments(data, w, wbar, m1, self.comm)

    def write_moments(self, data, label, w=None, wbar=None, m1=None):
        """Compute min, max, mean, and 2nd-6th central
        moments for assigned spatial field"""
        m = tcstats.central_moments(data, w, wbar, m1, self.comm)

        if self.comm.rank == 0:
            with open(self.moments_file, 'a') as fh:
                fmt = ('{:40s}  %s\n' % '  '.join(['{:14.8e}']*len(m))).format
                fh.write(fmt(label, *m))

        return m

    def mean(self, data, w=None, wbar=None):
        N = data.size

        if w is None:
            m1 = self.comm.allreduce(self.psum(data), op=MPI.SUM)/N
        else:
            if wbar is None:
                wbar = self.comm.allreduce(self.psum(w), op=MPI.SUM)/N
            N *= wbar
            m1 = self.comm.allreduce(self.psum(w*data), op=MPI.SUM)/N

        return m1

    def rms(self, data, w=None, wbar=None):
        """
        Spatial root-mean-square (RMS)
        """
        N = data.size

        if w is None:
            m2 = self.comm.allreduce(self.psum(data**2), op=MPI.SUM)/N
        else:
            if wbar is None:
                wbar = self.comm.allreduce(self.psum(w), op=MPI.SUM)/N
            N *= wbar
            m2 = self.comm.allreduce(self.psum(w*data**2), op=MPI.SUM)/N

        return m2**0.5

    def min_max(self, data):
        gmin = self.comm.allreduce(data.min(), op=MPI.MIN)
        gmax = self.comm.allreduce(data.max(), op=MPI.MAX)

        return (gmin, gmax)

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------
    def histogram1(self, var, fname, header='#', range=None, bins=100, w=None):
        """MPI-distributed univariate spatial histogram."""

        results = tcstats.histogram1(var, bins, range, w, self.comm)
        hist, lhist, bins, gmin, gmax = results

        # write histogram from root task
        if self.comm.rank == 0:
            fh = open('%s/%s-%s.hist' % (self.odir, self.prefix, fname), 'w')
            fh.write('%s\n' % header)
            fh.write('%d  %14.8e  %14.8e\n' % (bins, gmin, gmax))
            hist.tofile(fh, sep='\n', format='%14.8e')
            fh.close()

        return hist, gmin, gmax, lhist

    def histogram2(self, var1, var2, fname, header='#', range=None, bins=100,
                   w=None):
        """MPI-distributed bivariate spatial histogram."""

        if w is not None:
            w = np.ravel(w)

        var1, var2 = np.ravel(var1), np.ravel(var2)

        hist, lhist, bins1, bins2, range = tcstats.histogram2(
                                    var1, var2, range, bins, w, self.comm)

        # write histogram from root task
        if self.comm.rank == 0:
            save_file = '%s/%s.npz' % (self.odir, fname)
            np.savez(save_file, hist=hist, xy_range=range,
                     xbins=bins1, ybins=bins2, meta=header)

        return hist, range, lhist

    def conditional_mean(self, var, cond, fname, header=None,
                         bins=100, range=None, counts=None):
        """Computes the MPI-distributed mean of `var` conditional on
        the binned values of `cond`.

        This is an MPI-distributed equivalent to calling
        `scipy.stats.binned_static(cond, var, 'mean', bins, range)`.

        Parameters
        ----------
        var : MPI-distributed ndarray
            [description]
        cond : MPI-distributed ndarray
            [description]
        fname : string
            [description]
        header : string
            [description]
        bins : sequence or int, optional
            An array_like sequence describing the bin edges or an integer
            value describing the number of bins between the lower and upper
            range values (the default is 100).
        range : sequence, optional
            A sequence of lower and upper bin edges to be used if the edges
            are not given explicitly in `bins`. The default is to use the
            MPI-reduced minimum and maximum values of `cond`.
        counts : [N+2,]-shaped ndarray, optional
            (All ranks) the MPI-reduced histogram of `cond`, including
            outlier bins, generated, for instance, by a previous invocation
            of `conditional_mean`. If present, assumes `cond` is properly
            digitized, as if it were the `binned_cond` result of a previous
            invocation of `conditional_mean`.

        Returns
        -------
        cond_mean : [N,]-shaped ndarray
            (All ranks) ndarray of MPI-reduced conditional means of
            `var` with length N equal to the number of bins used to
            digitize `cond`.
        binned_cond : MPI-distribued ndarray
            (local to rank) digitized (with rightmost-edge correction)
            `cond` array.
        counts : [N+2,]-shaped ndarray
            (All ranks) MPI-reduced histogram of `cond`, including
            outlier bins.
        outliers : (2, )-shaped tuple
            (All ranks) MPI-reduced mean of `var` conditioned on low
            and high value of `cond` outside of the provided `bin` edge
            sequence or `range`.
        """
        if header is None:
            header = 'NOTE: data includes outlier bins.\n# cols: sums, counts'

        if counts is None:
            results = tcstats.conditional_mean(var, cond, bins, range,
                                               self.comm)
            cond_mean, binned_cond, counts, outliers = results
            binsum = np.r_[outliers[0], cond_mean, outliers[1]] * counts

            if np.isscalar(bins):
                nbin = bins
            else:
                nbin = len(bins) - 1

        else:
            # taken straight from tcstats.conditional_mean
            binned_cond = cond
            valid = counts.nonzero()
            nbin = counts.size

            binsum = np.bincount(binned_cond.ravel(), var.ravel(),
                                 minlength=nbin)
            self.comm.Allreduce(MPI.IN_PLACE, binsum, op=MPI.SUM)

            cond_mean = np.zeros(nbin, float)
            cond_mean.fill(np.nan)
            cond_mean[valid] = binsum[valid]/counts[valid]
            outliers = (cond_mean[0], cond_mean[-1])
            cond_mean = cond_mean[1:-1]

        if range is None:
            cmin = self.comm.allreduce(cond.min(), op=MPI.MIN)
            cmax = self.comm.allreduce(cond.max(), op=MPI.MAX)
        else:
            cmin, cmax = range

        # write conditional means to file from root task
        if (self.comm.rank == 0) and fname:
            with open(fname, 'w') as fh:
                fh.write('# %s\n' % header)
                fh.write('# %d %-14.8e %-14.8e\n' % (nbin, cmin, cmax))
                cols = np.zeros(counts.size, dtype=[
                                ('col1', np.float64), ('col2', np.int64)])
                cols['col1'] = binsum
                cols['col2'] = counts
                columns = np.stack((binsum, counts), axis=-1)
                np.savetxt(fh, columns, fmt='%-15.8e %-15d')

        return cond_mean, binned_cond, counts, outliers

    # -------------------------------------------------------------------------
    # Scalar and Vector Derivatives
    # -------------------------------------------------------------------------
    def div(self, var):
        """
        Calculate and return the divergence of a vector field.

        Currently the slab_exchange routines limit this function to vector
        fields.
        """
        div = self.deriv(var[0], dim=0)   # axis=2
        div+= self.deriv(var[1], dim=1)   # axis=1
        div+= self.deriv(var[2], dim=2)   # axis=0
        return div

    def curl(self, var):
        """
        Calculate and return the curl of a vector field.
        """
        assert var.ndim == 4, "ERROR: var must be a vector field!"

        temp = self.deriv(var[1], dim=2)
        omega = np.empty((3, *temp.shape), dtype=var.dtype)

        omega[0] = temp - self.deriv(var[2], dim=1)
        omega[1] = self.deriv(var[2], dim=0) - self.deriv(var[0], dim=2)
        omega[2] = self.deriv(var[0], dim=1) - self.deriv(var[1], dim=0)

        return omega

    def scl_grad(self, var):
        """
        Calculate and return the gradient vector field of a scalar field.
        """
        temp = self.deriv(var, dim=0)
        grad = np.empty((3, *temp.shape), dtype=var.dtype)

        grad[0] = temp
        grad[1] = self.deriv(var, dim=1)
        grad[2] = self.deriv(var, dim=2)

        return grad

    def grad(self, var):
        """
        Calculate and return the gradient tensor field of a vector field.
        """
        temp = self.deriv(var[0], dim=0)
        A = np.empty((3, 3, *temp.shape), dtype=var.dtype)

        for j in range(3):
            for i in range(3):
                A[j, i] = self.deriv(var[i], dim=j)

        return A

    def grad_curl_div(self, u):
        """
        Uses numpy.einsum which can be dramatically faster than
        alternative routines for many use cases
        """
        A = self.grad(u)

        e = np.zeros((3, 3, 3))
        e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
        e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
        omega = np.einsum('ijk,jk...->i...', e, A)

        Aii = np.einsum('ii...', A)

        return A, omega, Aii

    # -------------------------------------------------------------------------
    # Data Communication
    # -------------------------------------------------------------------------
    def exchange_ghost_cells(self, var):
        """
        Exchange ghost cells, including edge and corner cells, along
        MPI domain decomposition boundaries. Currently written to
        discover the number of ghost cells from the difference
        between var.shape and self.nnx, so the user must correctly size
        var.
        """
        nnx = self.nnx
        comm = self.comm
        coords = self.coords

        ND_shape = list(var.shape[-self.ndims:])
        extra_dims = var.ndim - self.ndims

        ng = (ND_shape - nnx)//2
        ixe = nnx + ng

        # array[all_slice] == array[:]
        all_slice = [slice(None, None, None)]*var.ndim

        # displacement along spatial dimension
        for disp in [-1, 1]:
            # spatial dimension
            for dim in range(extra_dims, var.ndim):
                i = dim - extra_dims

                disp_vec = np.zeros(self.ndims)
                disp_vec[dim] = disp
                send_rank = comm.Get_cart_rank(coords + disp_vec)
                recv_rank = comm.Get_cart_rank(coords - disp_vec)

                send_slice = list(all_slice)
                recv_slice = list(all_slice)

                # [ng, 2*ng) <-> [nnx+ng, nnx+2*ng)
                if disp == -1:
                    send_slice[dim] = slice(ng[i], 2*ng[i])
                    recv_slice[dim] = slice(ixe[i], None)
                # [0, ng) <-> [nnx, nnx+ng)
                else:
                    send_slice[dim] = slice(nnx[i], ixe[i])
                    recv_slice[dim] = slice(0, ng[i])

                send_slice = tuple(send_slice)
                recv_slice = tuple(recv_slice)

                gshape = ND_shape.copy()
                gshape[dim] = ng

                recv_ghost_cells = np.empty(gshape, dtype=np.float32)
                irecv_request = comm.Irecv(recv_ghost_cells, recv_rank, dim)

                send_ghost_cells = np.ascontiguousarray(var[send_slice])
                comm.Send(send_ghost_cells, send_rank, dim)

                irecv_request.wait()
                var[recv_slice] = recv_ghost_cells

        recv_ghost_cells = send_ghost_cells = None

        return var

    def z2y_slab_exchange(self, var):
        """
        Domain decomposition 'transpose' of MPI-distributed scalar array.
        Assumes 1D domain decomposition
        """

        nnz, ny, nx = var.shape
        nz = nnz*self.comm.size
        nny = ny//self.comm.size

        temp1 = np.empty([self.comm.size, nnz, nny, nx], dtype=var.dtype)
        temp2 = np.empty([self.comm.size, nnz, nny, nx], dtype=var.dtype)

        temp1[:] = np.rollaxis(var.reshape([nnz, self.comm.size, nny, nx]), 1)
        self.comm.Alltoall(temp1, temp2)  # send, receive
        temp2.resize([nz, nny, nx])

        return temp2

    def y2z_slab_exchange(self, varT):
        """
        Domain decomposition 'transpose' of MPI-distributed scalar array.
        Assumes 1D domain decomposition
        """

        nz, nny, nx = varT.shape
        nnz = nz//self.comm.size
        ny = nny*self.comm.size

        temp1 = np.empty([self.comm.size, nnz, nny, nx], dtype=varT.dtype)
        temp2 = np.empty([self.comm.size, nnz, nny, nx], dtype=varT.dtype)

        temp1[:] = varT.reshape(temp1.shape)
        self.comm.Alltoall(temp1, temp2)  # send, receive
        temp1.resize([nnz, ny, nx])
        temp1[:] = np.rollaxis(temp2, 1).reshape(temp1.shape)

        return temp1

    # -------------------------------------------------------------------------
    # Underlying Differential Operator Methods
    # -------------------------------------------------------------------------
    def _centdiff_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the specified derivative of a 3D scalar field at
        the specified order of accuracy.
        While k is passed on to the central_deriv function in the teslacu
        finite difference module, central_deriv may not honor a request for
        anything but the first derivative, depending on it's state of
        development.

        dim = dim % 3
        axis = 2-dim
        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = tcfd.central_deriv(var, self.dx[axis], bc='periodic',
                                   k=k, order=4, axis=axis)
        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv
        """
        pass

    def _akima_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the _first_ derivative of a 3D scalar field.
        The k parameter is ignored, a first derivative is _always_ returned.
        """
        dim = dim % 3
        axis = 2-dim
        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = tcas._deriv(var, self.dx[axis], axis=axis)

        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv

    def _akima_slab_deriv(self, var, dim=0, k=1, ng=3, valid_ghosts=False):
        """
        Calculate and return the _first_ derivative of a 3D scalar field.
        The k parameter is ignored, a first derivative is _always_ returned.
        """
        dim = dim % 3
        axis = 2-dim

        if axis == 0:
            var = self.z2y_slab_exchange(var)

        if not valid_ghosts:
            axes = (axis, (axis+1) % 3, (axis+2) % 3)
            shape = list(var.shape)
            shape[axis] += 2*ng
            temp = np.empty(shape, dtype=var.dtype)
            tempT = temp.transpose(axes)  # new _view_ into the inputs
            varT = var.transpose(axes)    # new _view_ into the inputs
            assert np.may_share_memory(temp, tempT)
            tempT[3:-3] = varT
            tempT[:3] = varT[-6:-3]
            tempT[-3:] = varT[3:6]
            var = temp

        deriv = tcas.flux_diff(var, dx=self.dx[axis], axis=axis, ng=3)

        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv


# -----------------------------------------------------------------------------
class _hitAnalyzer(_baseAnalyzer):
    """
    class _hitAnalyzer(_baseAnalyzer) ...
    Empty Docstring!
    """

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, comm, odir, pid, ndims, decomp, L, N, method):

        periodic = [True]*ndims
        super().__init__(comm, odir, pid, ndims, decomp, periodic, L, N,
                         'ignore')

        self._config = "Homogeneous Isotropic Turbulence"

        if self._decomp == 1 and ndims == 3:
            # Spectral variables (1D Decomposition, 3D field)
            self.nk = self.nx.copy()
            self.nk[2] = self.nx[2]//2+1
            self.nnk = self.nk.copy()
            self.nnk[1] = self.nk[1]//comm.size
            self.dk = 2*np.pi/self.L[2]

            nx = self.nx[2]
            dk = self.dk

            nny = self.nnk[1]
            iys = nny*comm.rank
            iye = iys + nny

            # The teslacu.fft.rfft3 and teslacu.fft.irfft3 functions currently
            # transpose Z and Y in the forward fft (rfft3) and inverse the
            # tranpose in the inverse fft (irfft3).
            # These FFT routines and these variables below assume that ndims=3
            # which ruins the generality I so carefully crafted in the base
            # class
            k2 = np.fft.rfftfreq(self.nx[2])*nx * dk
            k1 = np.fft.fftfreq(self.nx[1])*nx * dk
            k1 = k1[iys:iye].copy()
            k0 = np.fft.fftfreq(self.nx[0])*nx * dk

            # MPI local 3D wavemode index
            self.Kvec = np.array(np.meshgrid(k0, k1, k2, indexing='ij'))
            self.Kmag = np.sqrt(np.sum(np.square(self.Kvec), axis=0))
            self.Kmode = (self.Kmag//dk).astype(int)
            self.k = k2

        if method == 'akima_flux_diff':
            self.deriv = self._akima_slab_deriv
        elif method == 'central_diff':
            self.deriv = self._centdiff_deriv
        elif method == 'spectral':
            self.deriv = self._fft_deriv
        else:
            if comm.rank == 0:
                print("mpiAnalyzer._hitAnalyzer.__init__(): "
                      "'method' argument not recognized!\n"
                      "Defaulting to Akima spline flux differencing.")
            self.deriv = self._akima_slab_deriv

    # -------------------------------------------------------------------------
    # Power Spectral Density Analysis
    # -------------------------------------------------------------------------
    def spectral_density(self, var, fname, metadata=''):
        """
        Write the 1D power spectral density of var to text file. Method
        assumes a real input is in physical space and a complex input is
        in Fourier space.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        if np.iscomplexobj(var) is True:
            cdata = var
        else:
            if var.ndim == 3:
                cdata = tcfft.rfft3(self.comm, var)
            elif var.ndim == 4:
                cdata = self.vec_fft(var)
            else:
                raise AttributeError('Input is {}D, '.format(var.ndim)
                                     +'spectral_density expects 3D or 4D!')

        # get spectrum (each task will receive the full spectrum)
        spect3d = np.real(cdata*np.conj(cdata))
        if var.ndim == 4:
            spect3d = np.sum(spect3d, axis=0)
        spect3d[..., 0] *= 0.5

        spect1d = tcfft.shell_average(self.comm, spect3d, self.Kmode)

        if self.comm.rank == 0:
            fh = open('%s/%s-%s.spectra' % (
                                    self.odir, self.prefix, fname), 'w')
            fh.write('%s\n' % metadata)
            spect1d.tofile(fh, sep='\n', format='% .8e')
            fh.close()

        return spect1d

    def integral_scale(self, Ek):
        """
        Computes the integral scale from the standard formula,
        where u'^2 = 2/3*Int{Ek}
        ell = (pi/2)*(1/u'^2)*Int{Ek/k}
            = 3*pi/4*Int{Ek/k}/Int{Ek}
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return 0.75*np.pi*self.psum(Ek[1:]/self.k[1:])/self.psum(Ek[1:])

    def shell_average(self, E3):
        """
        Convenience function for shell averaging
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.shell_average(self.comm, E3, self.Kmode)

    def filter_kernel(self, ell, gtype='gaussian'):
        """
        Empty docstring!
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        if gtype == 'tophat':
            # THIS IS WRONG STILL, NOT 3D!!!
            Ghat = np.sinc(self.Kmag*ell)

        elif gtype == 'gaussian':
            Ghat = np.exp(-0.5*(np.pi*ell*self.Kmag)**2)

        elif gtype == 'comp_exp':
            """
            A 'COMPact EXPonential' filter which has
            1) compact support in a ball of radius ell (or 1/ell)
            2) is strictly positive, and
            3) is smooth (infinitely differentiable)
            in _both_ physical and spectral space!
            """
            kl = self.Kmag*ell
            Ghat = np.where(kl <= 0.5, np.exp(-kl**2/(0.25-kl**2)), 0.0)
            G = tcfft.irfft3(self.comm, Ghat.astype(np.complex128))
            G = G**2
            Gbar = self.comm.allreduce(self.psum(G), op=MPI.SUM)
            G *= 1.0/Gbar
            Ghat = tcfft.rfft3(self.comm, G)

        elif gtype == 'spectral':
            Ghat = np.where(self.Kmag*ell < 1.0, 1.0, 0.0)

        else:
            raise ValueError('did not understand filter type')

        return Ghat

    def scalar_filter(self, phi, Ghat):
        """Filter scalar field with a Fourier-domain transfer function.

        Assumes `phi` is a 3D scalar spatial field with 1D domain
        decomposition. `phi` is transformed to Fourier-domain, multiplied
        by `Ghat`, then inverse transformed back to spatial domain.

        Parameters
        ----------
        phi : {[M//nprocs, N, P]-shaped array_like}
            3D spatial field to be filtered.
        Ghat : {[M, N//nprocs, P//2+1]-shaped array_like}
            3D Fourier-domain filter transfer function.

        Returns
        -------
        [M//nprocs, N, P]-shaped ndarray
            Filtered scalar field.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.irfft3(self.comm, Ghat*tcfft.rfft3(self.comm, phi))

    def vector_filter(self, u, Ghat):
        assert self._decomp == 1, "ERROR decomp not 1D"

        return self.vec_ifft(Ghat*self.vec_fft(u))

    # -------------------------------------------------------------------------
    # FFT Wrapper Methods
    # -------------------------------------------------------------------------

    def _fft_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the specified derivative of a 3D scalar field.
        This function uses 1D FFTs and MPI-decomposed transposing instead of
        MPI-decomposed 3D FFTs.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        dim = dim % 3
        axis = 2-dim
        s = [1]*var.ndim
        s[axis] = self.k.shape[0]
        K = self.k.reshape(s)

        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = np.fft.irfft(
                    np.power(1j*K, k)*np.fft.rfft(var, axis=axis), axis=axis)

        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv

    def scl_fft(self, var):
        """
        Convenience function for MPI-distributed 3D r2c FFT of scalar.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.rfft3(self.comm, var)

    def scl_ifft(self, var):
        """
        Convenience function for MPI-distributed 3D c2r IFFT of scalar.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.irfft3(self.comm, var)

    def vec_fft(self, var):
        """
        Convenience function for MPI-distributed 3D r2c FFT of vector.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        nnz, ny, nx = var.shape[1:]
        nk = nx//2+1
        nny = ny//self.comm.size
        nz = nnz*self.comm.size

        if var.dtype.itemsize == 8:
            fft_complex = np.complex128
        elif var.dtype.itemsize == 4:
            fft_complex = np.complex64
        else:
            raise AttributeError("cannot detect dataype of u")

        fvar = np.empty([3, nz, nny, nk], dtype=fft_complex)
        fvar[0] = tcfft.rfft3(self.comm, var[0])
        fvar[1] = tcfft.rfft3(self.comm, var[1])
        fvar[2] = tcfft.rfft3(self.comm, var[2])

        return fvar

    def vec_ifft(self, fvar):
        """
        Convenience function for MPI-distributed 3D c2r IFFT of vector.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        nz, nny, nk = fvar.shape[1:]
        nx = (nk-1)*2
        ny = nny*self.comm.size
        nnz = nz//self.comm.size

        if fvar.dtype.itemsize == 16:
            fft_real = np.float64
        elif fvar.dtype.itemsize == 8:
            fft_real = np.float32
        else:
            raise AttributeError("cannot detect dataype of u")

        var = np.empty([3, nnz, ny, nx], dtype=fft_real)
        var[0] = tcfft.irfft3(self.comm, fvar[0])
        var[1] = tcfft.irfft3(self.comm, fvar[1])
        var[2] = tcfft.irfft3(self.comm, fvar[2])

        return var

    def half_reflect_periodization(self, var, izs=None):
        """Shift and reflect half of scalar volume in z-direction.

        MPI-transpose a 3D scalar volume with 1D domain decomposition
        then take center half (nz//4:3*nz//4), shift (to 0:nz//2), and
        reflect along z-direction (varT[:nz//2] == varT[nz//2::-1])

        Parameters
        ----------
        var : MPI-distributed ndarray
            MPI-decomposed 3D scalar volume (1D domain decomposition)

        Returns
        -------
        MPI-distributed ndarray
            reflected center half-volume
        """
        varT = self.z2y_slab_exchange(var)
        nz, nny, nx = varT.shape
        nzq = nz//4
        nzh = nz//2
        if izs is None:
            izs = nzq

        temp = np.empty((nzh, nny, nx), dtype=varT.dtype)
        temp[:] = varT[izs:izs+nzh]
        varT[:nzh] = temp
        varT[nzh:] = temp[::-1]

        assert np.all(varT[nzq] == varT[nzq+nzh-1])

        var[:] = self.y2z_slab_exchange(varT)

        return var

    def qtr_truncate_reflected_var(self, var, izs=None):
        # at the start var is reflected in center, so only half the domain
        # contains unique information. We want just half of that half.
        varT = self.z2y_slab_exchange(var)

        nz, nny, nx = varT.shape
        izs = nz//8
        ize = izs+nz//4

        return self.y2z_slab_exchange(varT[izs:ize])

    def truncate_along_z(self, var, izs, ize):
        varT = self.z2y_slab_exchange(var)
        return self.y2z_slab_exchange(varT[izs:ize])

    def flame_extents(self, Y):
        YT = self.z2y_slab_exchange(Y)

        zfs = np.min(np.nonzero(YT > 0.001)[0])
        zfe = np.max(np.nonzero(YT < 0.999)[0])

        zfs = self.comm.allreduce(zfs, op=MPI.MIN)
        zfe = self.comm.allreduce(zfe, op=MPI.MAX)

        return zfs, zfe
