"""
"""

from mpi4py import MPI
import numpy as np
# from math import sqrt
import time
import sys
import argparse

__all__ = []


class LoadInputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        this overloads the template argparse.Action.__call__ method
        and must keep the same argument names
        """

        # argument values is an open file handle, with statement is just added
        # prettiness to make it clear that values is a file handle
        with values as fh:
            raw_lines = fh.read().splitlines()
            fh.close()

        stripped_lines = [rl.strip().split('#')[0] for rl in raw_lines]
        # the following strip blank lines, since '' evaluates to False
        input_lines = [sl for sl in stripped_lines if sl]
        args = []

        for line in input_lines:
            if line.startswith('<'):
                # These file lines are considered special just in case
                # we ever want to do something with these headers, like
                # spit the whole simulation config back out to stdout/log
                # like Athena does.
                #
                # line.strip()[1:-1]
                pass
            else:
                key, val = [kv.strip() for kv in line.strip().split('=')]
                args.append('--%s=%s' % (key, val))

        parser.parse_known_args(args, namespace)

        return


def timeofday():
    return time.strftime("%H:%M:%S")


def scalar_analysis(mA, phi, range, m1, w, wbar, fname, title, symb, norm=1.):
    """
    Compute all the 'stand-alone' statistics and scalings related to
    a scalar field such as density, pressure, or tracer mass
    fraction.

    Arguments
    ---------
    mA    : mpiAnalyzer object
    phi   : scalar data field
    w     : scalar weights field
    wbar  : mean of weights field
    fname : file name string
    title : written name of phi
    symb  : latex math string symbolizing phi
    """

    # update = '{:4d}\t{}'.format

    if w is None:
        ylabel = "\mathrm{pdf}"
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
    else:
        ylabel = "\widetilde{\mathrm{pdf}}"
        xlabel = "%s\t\left\{{{}\\right\}}" % symb

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    mA.mpi_histogram1(phi, fname, '%s\t%s' % (xlabel, ylabel),
                      range, 100, w, wbar, m1, norm)

    mA.write_mpi_moments(phi, title, w, wbar, m1, norm)

    if fname in ['rho', 'P', 'T', 'Smm', 'Y']:
        mA.spectral_density(phi, fname, '%s PSD\t%s' % (title, Ek_fmt(symb)))

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def vector_analysis(mA, v, minmax, m1, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics and scalings related to
    a vector field such as velocity, momentum, or vorticity.

    Arguments
    ---------
    v     : vector field (1st order tensor)
    fname : file name string
    symb  : latex math string
    """

    if w is None:
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
        ylabel = "\mathrm{pdf}"
        wbar = None
        wvec = None
    else:
        xlabel = "%s\t\left\{{{}\\right\}}" % symb
        ylabel = "\widetilde{\mathrm{pdf}}"
        if w.size == v.size/v.shape[0]:
            s = [1]*w.ndim
            s.insert(0, v.shape[0])
            wvec = np.tile(w, s)
        elif w.size == v.size:
            wvec = w
        else:
            raise ValueError("w should either be the same size as v or"+
                             "the same size as one component of v")

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # vector components analyzed
    mA.mpi_histogram1(v.copy(), fname, xlabel, ylabel, minmax,
                      100, wvec, wbar, norm=3.0)
    mA.spectral_density(v, fname, title+' PSD', Ek_fmt(symb))
    mA.write_mpi_moments(v, title, symb, wvec, wbar, m1=m1, norm=3.0)

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def gradient_analysis(mA, A, minmax, m1, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics of the velocity-
    gradient tensor field.

    Arguments
    ---------
    A     : velocity gradient tensor field (2nd order)
    fname : file name string
    symb  : latex math string
    """

    for j in range(0, 3):
        for i in range(0, 3):
            tij = ' {}{}'.format(i+1, j+1)
            sij = '_{{{}{}}}'.format(i+1, j+1)
            mA.write_mpi_moments(
                            A[j, i], title+tij, symb+sij, w, wbar, m1=m1)

    if w is None:
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
        ylabel = "\mathrm{pdf}"
        wbar = None
        W = None
    else:
        xlabel = "%s\t\left\{{{}\\right\}}" % symb
        ylabel = "\widetilde{\mathrm{pdf}}"
        if w.size == A.size/(A.shape[0]*A.shape[1]):
            s = [1]*w.ndim
            s.insert(0, A.shape[1])
            s.insert(0, A.shape[0])
            W = np.tile(w, s)
        elif w.size == A.size:
            W = w
        else:
            raise ValueError("w should either be the same size as v or"
                             "the same size as one component of v")

    symb += '_{ij}'

    mA.mpi_histogram1(A.copy(), fname, xlabel, ylabel, minmax, 100, W, wbar,
                      norm=9.0)

    # Aii = np.einsum('ii...', A)
    # I = np.identity(3)/3.0
    # s = np.ones(len(Aii.shape)+2, dtype=np.int)
    # s[0:2] = 3
    # Atl = A-I.reshape(s)*Aii
    # m1 = (m1.sum()-m1[0, 0]-m1[1, 1]-m1[2, 2])/9.0

    # symb += "^\mathrm{tl}"
    # fname += '_tl'
    # mA.mpi_histogram1(Atl, fname, xlabel, ylabel, 100, W, wbar,
    #                   m1, 9.0)

    # add tensor invariants here.

    return None  # add success flag or error
