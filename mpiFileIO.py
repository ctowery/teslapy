"""
Description:
============
This module contains the mpiFileIO object classes for the TESLaCU package.
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

Coding Style Guide:
-------------------
This module generally adheres to the Python style guide published in
PEP 8, with the following exceptions:
- Warning W503 (line break occurred before a binary operator) is
  ignored, since this warning is a mistake and PEP 8 recommends breaking
  before operators
- Error E225 (missing whitespace around operator) is ignored

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Definitions:
============

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
import os

__all__ = ['mpiFileIO', '_binaryFileIO']


def mpiFileIO(comm=MPI.COMM_WORLD, idir='./', odir='./', ftype=None,
              ndims=3, N=[512]*3, decomp=None, **kwargs):
    """
    The mpiFileIO() function is a "class factory" which returns the
    appropriate mpi-parallel reader class instance based upon the
    inputs. Each subclass contains a different ... (finish docstring)

    Arguments:

    Output:
    """

    if ftype in ['binary', None]:
        periodic = kwargs.pop('periodic', None)
        ixs = kwargs.pop('ixs', None)
        ixe = kwargs.pop('ixe', None)
        newFileIO = _binaryFileIO(comm, idir, odir, ndims, N, decomp, periodic,
                                  ixs, ixe)
    else:
        raise ValueError('unknown ftype!')

    return newFileIO
# -----------------------------------------------------------------------------


class _binaryFileIO(object):
    """
    Empty docstring!

    binaryFileIO.__init__() could be moved to (or used as) a base FileIO class
    """

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, comm, idir, odir, ndims, N, decomp, periodic, ixs, ixe):

        self.idir = idir  # users should be able to change directories at will
        self.odir = odir  # users should be able to change directories at will

        self._ndims = ndims
        self._subarrays = dict()
        init_comm = comm

        if np.iterable(N):
            if len(N) == 1:
                self._nxf = np.array(list(N)*ndims, dtype=int)
            elif len(N) == ndims:
                self._nxf = np.array(N, dtype=int)
            else:
                raise IndexError("The length of N must be either 1 or ndims")
        else:
            self._nxf = np.array([N]*ndims, dtype=int)
        nxf = self._nxf

        if decomp is None:
            self._decomp = 1
        elif decomp in [1, 2, 3] and decomp <= ndims:
            self._decomp = decomp
        else:
            raise IndexError("decomp must be 1, 2, 3 or None")

        if periodic is None:
            self._periodic = tuple([True]*ndims)
        elif len(periodic) == ndims:
            self._periodic = tuple(periodic)
        else:
            raise IndexError("Either len(periodic) must be ndims or "
                             "periodic must be None")

        if ixs is not None:
            if len(ixs) == 1:
                ixs = np.array(list(ixs)*ndims, dtype=int)
            elif len(ixs) == ndims:
                ixs = np.array(ixs, dtype=int)
            else:
                raise IndexError("The length of ixs must be either 1 or ndims")
        else:
            ixs = np.zeros(ndims, dtype=np.int32)

        if ixe is not None:
            if len(ixe) == 1:
                ixe = np.array(list(ixe)*ndims, dtype=int)
            elif len(ixe) == ndims:
                ixe = np.array(ixe, dtype=int)
            else:
                raise IndexError("The length of ixe must be either 1 or ndims")
        else:
            ixe = nxf

        dims = MPI.Compute_dims(init_comm.size, self._decomp)
        dims.extend([1]*(ndims-self._decomp))
        self._comm = init_comm.Create_cart(dims, self._periodic)

        self._nx = ixe - ixs
        self._nnx = self._nx//dims
        self._ixs = ixs + self._nnx*self.comm.coords
        self._ixe = self._ixs + self._nnx

        assert np.all(np.mod(self._nxf, dims) == 0)

    # -------------------------------------------------------------------------
    # Class Properities
    # -------------------------------------------------------------------------
    def __enter__(self):
        # with statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with statement finalization
        for key, (etype, filetype) in self._subarrays.items():
            filetype.Free()
        self._comm.Disconnect()
        return False

    @property
    def comm(self):
        return self._comm

    @property
    def ndims(self):
        return self._ndims

    @property
    def decomp(self):
        return self._decomp

    @property
    def periodic(self):
        return self._periodic

    @property
    def nxf(self):
        return self._nxf

    @property
    def nx(self):
        return self._nx

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
    # Class Methods
    # -------------------------------------------------------------------------

    def Read_all(self, filename, ftype=np.float32, mtype=np.float32):
        """
        Read an entire file into MPI-distributed memory with n-dimensional
        decomposition specified during construction of mpiFileIO object.
        """
        ftype = np.dtype(ftype)
        mtype = np.dtype(mtype)

        key = '%s%d' % (ftype.kind, ftype.itemsize)
        if key in self._subarrays:
            etype, filetype = self._subarrays[key]
        else:
            etype, filetype = self._Create_subarray(key)

        status = MPI.Status()
        temp = np.zeros(self._nnx, dtype=ftype)

        fh = MPI.File.Open(self.comm, os.path.join(self.idir, filename))

        # adding a loop over disp would allow for files with multiple scalars
        disp = 0
        fh.Set_view(disp, etype, filetype, 'native')
        fh.Read_all(temp, status)

        fh.Close()

        return temp.astype(mtype, casting='safe')

    def Write_all(self, filename, data, ftype=np.float32):
        """
        Write an entire file into MPI-distributed memory with n-dimensional
        decomposition specified during construction of mpiFileIO object.
        """
        assert np.all(np.array(data.shape) == self._nnx)

        filename = os.path.join(self.odir, filename)
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            if self.comm.rank == 0:
                os.makedirs(dirname)

        ftype = np.dtype(ftype)
        temp = data.astype(ftype, casting='safe', copy=False)

        key = '%s%d' % (ftype.kind, ftype.itemsize)
        if key in self._subarrays:
            etype, filetype = self._subarrays[key]
        else:
            etype, filetype = self._Create_subarray(key)

        status = MPI.Status()
        fh = MPI.File.Open(self.comm, filename,
                           MPI.MODE_WRONLY | MPI.MODE_CREATE)

        # adding a loop over disp would allow for multiple scalars per file
        disp = 0
        fh.Set_view(disp, etype, filetype, 'native')
        fh.Write_all(temp, status)

        fh.Close()

        return status

    def _Create_subarray(self, key):
        """Private module function.

        [description]

        Parameters
        ----------
        key : {[type]}
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """

        if key == 'f4':
            etype = MPI.REAL4
        elif key == 'f8':
            etype = MPI.REAL8
        elif key == 'i4':
            etype = MPI.INTEGER4
        elif key == 'i8':
            etype = MPI.INTEGER8
        elif key == 'c8':
            etype = MPI.COMPLEX8
        elif key == 'c16':
            etype = MPI.COMPLEX16
        elif key == 'f16':
            etype = MPI.FLOAT16
        else:
            raise ValueError("etype not supported by mpiFileIO. End users can"
                             " modify this if/elif structure with the"
                             " required MPI etype, if they know what it is.")

        filetype = etype.Create_subarray(self._nxf, self._nnx, self._ixs)
        filetype.Commit()

        self._subarrays[key] = (etype, filetype)

        return etype, filetype

    def _Read_at_all_1d(self, filename, ftype=np.float32, mtype=np.float64):
        """
        A simpler function for reading in a file with 1D domain decomposition.
        Probably also faster, as well, but by how much needs to be measured.
        """
        temp = np.zeros(self.nnx, dtype=ftype)
        status = MPI.Status()
        fh = MPI.File.Open(self.comm, os.path.join(self.idir, filename))

        # adding a loop over offset would allow for multiple scalars per file
        offset = self.comm.rank*temp.nbytes
        fh.Read_at_all(offset, temp, status)

        fh.Close()

        return temp.astype(mtype, casting='safe', copy=False)

    def _Write_at_all_1d(self, filename, data, ftype=np.float32):
        """
        A simpler function for writing to a file with 1D domain decomposition.
        Probably also faster, as well, but by how much needs to be measured.
        """
        assert np.all(np.array(data.shape) == self._nnx)

        temp = data.astype(ftype, casting='safe', copy=False)
        status = MPI.Status()
        fh = MPI.File.Open(self.comm, os.path.join(self.odir, filename),
                           MPI.MODE_WRONLY | MPI.MODE_CREATE)

        # adding a loop over offset would allow for multiple scalars per file
        offset = self.comm.rank*temp.nbytes
        fh.Write_at_all(offset, temp, status)

        fh.Close()

        return status
