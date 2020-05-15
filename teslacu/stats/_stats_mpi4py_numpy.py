"""
MPI-distributed statistics methods. Does not require information about
domain decomposition or dimensionality.
"""

from mpi4py import MPI
import numpy as np

__all__ = ['psum', 'central_moments', 'conditional_mean',
           'histogram1', 'histogram2']

COMM_WORLD = MPI.COMM_WORLD


def psum(data):
    """
    input argument data can be any n-dimensional array-like object, including
    a 0D scalar value or 1D array.
    """

    psum = np.array(data)  # np.array always hard-copies
    for n in range(psum.ndim):
        psum.sort(axis=-1)
        psum = np.sum(psum, axis=-1)
    return psum


def std_dev(data, N=None, w=None, wbar=None, m1=None, comm=COMM_WORLD):
    """
    Empty Docstring!
    """

    gmin = comm.allreduce(np.nanmin(data), op=MPI.MIN)
    gmax = comm.allreduce(np.nanmax(data), op=MPI.MAX)

    if w is None:   # unweighted moments
        w = 1.0
        wbar = 1.0
    else:
        assert w.size == data.size

    if wbar is None:
        wbar = comm.allreduce(psum(w), op=MPI.SUM)/w.size

    if N is None:
        N = data.size

    N_inv = 1.0/(N*wbar)

    # 1st (raw) moment
    if m1 is None:
        m1 = comm.allreduce(psum(w*data), op=MPI.SUM)*N_inv

    cdata = data - m1
    c2 = comm.allreduce(psum(w*np.power(cdata, 2)), op=MPI.SUM)*N_inv
    std = c2**0.5

    return std, m1, gmin, gmax


def central_moments(data, N=None, w=None, wbar=None, m1=None, comm=COMM_WORLD):
    """
    Computes global min, max, and 1st to 6th central moments of
    MPI-distributed data.

    To get raw moments, simply pass in m1=0.
    To get weighted moments, bass in the weighting coefficients, w.
    If you've already calculated the mean of w, save some computations and pass
    it in as wbar.
    """

    gmin = comm.allreduce(np.nanmin(data), op=MPI.MIN)
    gmax = comm.allreduce(np.nanmax(data), op=MPI.MAX)

    if w is None:   # unweighted moments
        w = 1.0
        wbar = 1.0
    else:
        assert w.size == data.size

    if wbar is None:
        wbar = comm.allreduce(psum(w), op=MPI.SUM)/w.size

    if N is None:
        N = data.size

    N_inv = 1.0/(N*wbar)

    # 1st raw moment
    if m1 is None:
        m1 = comm.allreduce(psum(w*data), op=MPI.SUM)*N_inv

    # 2nd-4th centered moments
    cdata = data - m1
    c2 = comm.allreduce(psum(w*np.power(cdata, 2)), op=MPI.SUM)*N_inv
    c3 = comm.allreduce(psum(w*np.power(cdata, 3)), op=MPI.SUM)*N_inv
    c4 = comm.allreduce(psum(w*np.power(cdata, 4)), op=MPI.SUM)*N_inv
    c5 = comm.allreduce(psum(w*np.power(cdata, 5)), op=MPI.SUM)*N_inv
    c6 = comm.allreduce(psum(w*np.power(cdata, 6)), op=MPI.SUM)*N_inv

    return m1, c2, c3, c4, c5, c6, gmin, gmax


def conditional_mean(var, cond, bins=100, range=None, comm=COMM_WORLD):
    """Computes the MPI-distributed mean of `var` conditional on the
    binned values of `cond`.

    This is an MPI-distributed equivalent to calling
    `scipy.stats.binned_static(cond, var, 'mean', bins, range)`.

    Parameters
    ----------
    var : MPI-distributed ndarray
        [description]
    cond : MPI-distributed ndarray
        [description]
    bins : sequence or int, optional
        An array_like sequence describing the bin edges or an integer
        value describing the number of bins between the lower and upper
        range values (the default is 100).
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges
        are not given explicitly in `bins`. The default is to use the
        MPI-reduced minimum and maximum values of `cond`.
    comm : MPI.Comm, optional
        MPI communicator over which `var` and `cond` are distrbuted.
        (Default is MPI.COMM_WORLD)

    Returns
    -------
    cond_mean : [N,]-shaped ndarray
        (All ranks) ndarray of MPI-reduced conditional mean of `var`
        with length N equal to the number of bins used to digitize `cond`.
    binned_cond : MPI-distribued ndarray
        (local to rank) digitized and rightmost-edge-corrected `cond` array.
    counts : [N+2,]-shaped ndarray
        (All ranks) MPI-reduced histogram of `cond`, including outlier bins.
    outliers : (2, )-shaped tuple
        (All ranks) MPI-reduced mean of `var` conditioned on low and
        high outliers of `cond` outside of provided `bin` edge sequence
        or `range`.
    """

    # Create edge arrays
    if np.isscalar(bins):
        nbin = bins + 2

        # Get the range
        if range is None:
            fill = np.ma.minimum_fill_value(cond)
            cmin = comm.allreduce(np.amin(cond, initial=fill), op=MPI.MIN)
            cmax = comm.allreduce(np.amax(cond, initial=-fill), op=MPI.MAX)
        else:
            cmin, cmax = range

        edges = np.linspace(cmin, cmax, nbin - 1)

    else:
        edges = np.asarray(bins, float)
        nbin = edges.size + 1

    dedges = np.diff(edges)

    # Compute the bin number each data point falls into
    binned_cond = np.digitize(cond, edges)

    # Using `digitize`, values that fall on an edge are put in the right
    # bin. For the rightmost bin, we want values equal to the right edge
    # to be counted in the last bin, and not as an outlier.
    # -- Find the rounding precision
    decimal = int(-np.log10(dedges.min())) + 6
    # -- Find which points are on the rightmost edge.
    on_edge = np.where(np.around(cond, decimal) ==
                       np.around(edges[-1], decimal))
    # -- Shift these points one bin to the left.
    binned_cond[on_edge] -= 1

    counts = np.bincount(binned_cond.ravel(), minlength=nbin)
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)

    cond_mean = np.empty(nbin, float)
    cond_mean.fill(np.nan)
    valid = counts.nonzero()
    binsum = np.bincount(binned_cond.ravel(), var.ravel(), minlength=nbin)
    comm.Allreduce(MPI.IN_PLACE, binsum, op=MPI.SUM)
    cond_mean[valid] = binsum[valid]/counts[valid]

    outliers = (cond_mean[0], cond_mean[-1])
    cond_mean = cond_mean[1:-1]

    return cond_mean, binned_cond, counts, outliers


def histogram1(var, bins=50, range=None, w=None, comm=COMM_WORLD):
    """
    Constructs the histogram (probability mass function) of MPI-
    distributed data. Now safe for null-sized arrays.
    """

    if range is None:
        fill = np.ma.minimum_fill_value(var)
        gmin = comm.allreduce(np.amin(var, initial=fill), op=MPI.MIN)
        gmax = comm.allreduce(np.amax(var, initial=-fill), op=MPI.MAX)
    else:
        (gmin, gmax) = range

    temp = np.histogram(var, bins=bins, range=(gmin, gmax), weights=w)[0]
    lhist = np.ascontiguousarray(temp)
    ghist = np.empty_like(lhist)
    comm.Allreduce(lhist, ghist, op=MPI.SUM)

    return ghist, lhist, bins, gmin, gmax


def histogram2(var1, var2, range=None, bins=50, w=None, comm=COMM_WORLD):
    """
    Constructs the 2D histogram (probability mass function) of two MPI-
    distributed data sets. Now safe for null-sized arrays.
    """
    if np.iterable(bins):
        bins1 = bins[0]
        bins2 = bins[1]
    else:
        bins1 = bins2 = bins

    if range is None:
        fill = np.ma.minimum_fill_value(var1)
        gmin1 = comm.allreduce(np.amin(var1, initial=fill), op=MPI.MIN)
        gmax1 = comm.allreduce(np.amax(var1, initial=-fill), op=MPI.MAX)

        fill = np.ma.minimum_fill_value(var2)
        gmin2 = comm.allreduce(np.amin(var2, initial=fill), op=MPI.MIN)
        gmax2 = comm.allreduce(np.amax(var2, initial=-fill), op=MPI.MAX)

        range = ((gmin1, gmax1), (gmin2, gmax2))
    else:
        ((gmin1, gmax1), (gmin2, gmax2)) = range

    temp = np.histogram2d(var1, var2, bins=bins, range=range, weights=w)[0]
    lhist = np.ascontiguousarray(temp)
    ghist = np.empty_like(lhist)
    comm.Allreduce(lhist, ghist, op=MPI.SUM)

    return ghist, lhist, bins1, bins2, range


# def alt_local_moments(data, w=None, wbar=None, N=None, unbias=True):
#     """
#     Returns the mean and 2nd-4th central moments of a memory-local
#     numpy array as a list. Default behavior is to return unbiased
#     sample moments for 1st-3rd order and a partially-corrected
#     sample 4th central moment.
#     """

#     if w is None:
#         u1 = psum(data)/N
#         if unbias:
#             c2 = psum(np.power(data-u1, 2))/(N-1)
#             c3 = psum(np.power(data-u1, 3))*N/(N**2-3*N+2)
#             c4 = psum(np.power(data-u1, 4))*N**2/(N**3-4*N**2+5*N-1)
#             c4+= (3/(N**2-3*N+3)-6/(N-1))*c2**2
#         else:
#             c2 = psum(np.power(data-u1, 2))/N
#             c3 = psum(np.power(data-u1, 3))/N
#             c4 = psum(np.power(data-u1, 4))/N
#     else:
#         if wbar is None:
#             wbar = psum(w)/N

#         u1 = psum(w*data)/(N*wbar)
#         if unbias:
#             c2 = psum(w*np.power(data-u1, 2))/(wbar*(N-1))
#             c3 = psum(w*np.power(data-u1, 3))*N/((N**2-3*N+2)*wbar)
#             c4 = psum(w*np.power(data-u1, 4))*N**2
#             c4/= (N**3-4*N**2+5*N-1)*wbar
#             c4+= (3/(N**2-3*N+3)-6/(N-1))*c2**2
#         else:
#             c2 = psum(w*np.power(data-u1, 2))/(N*wbar)
#             c3 = psum(w*np.power(data-u1, 3))/(N*wbar)
#             c4 = psum(w*np.power(data-u1, 4))/(N*wbar)

#     return u1, c2, c3, c4
