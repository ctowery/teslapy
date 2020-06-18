"""
"""
from mpi4py import MPI
import numpy as np
import argparse
# from memory_profiler import profile

from .mpiAnalyzer import _baseAnalyzer
from .mpiFileIO import mpiFileIO
from . import fft as tcfft

__all__ = []

COMM = MPI.COMM_WORLD


###############################################################################
class turbFlameAnalyzer(_baseAnalyzer):
    """
    class turbFlameAnalyzer(_baseAnalyzer) ...
    Empty Docstring!
    """

    parser = argparse.ArgumentParser(prog='turbFlameAnalyzer',
                                     add_help=False)
    parser.add_argument('--idir', type=str)
    parser.add_argument('--odir', type=str)
    parser.add_argument('--pid', type=str)
    parser.add_argument('--N', type=int)
    parser.add_argument('--L', type=float)
    parser.add_argument('--its', type=int)
    parser.add_argument('--ite', type=int)
    parser.add_argument('--tint', type=int)
    parser.add_argument('--ntm', type=int)
    parser.add_argument('--iz0', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--speed', type=float)
    parser.add_argument('--jfs', type=float)
    parser.add_argument('--jfe', type=float)
    parser.add_argument('--fint', type=float)

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, comm=COMM, L=np.pi, N=512, odir='./', pid='test'):

        ndims = 3
        decomp = 1
        periodic = [True, ]*3
        method = 'akima_flux_diff'

        super().__init__(comm, odir, pid, ndims, decomp, periodic, L, N,
                         method)

        self._config = "Coarse-graining analysis of `turbflame`-like data"

        # The teslacu.fft.rfft3 and teslacu.fft.irfft3 functions currently
        # transpose Z and Y in the forward fft (rfft3) and inverse the
        # tranpose in the inverse fft (irfft3).
        # These FFT routines and these variables below assume that ndims=3
        # which ruins the generality I so carefully crafted in the base
        # mpiAnalyzer class
        self.nk = self.nx.copy()
        self.nk[2] = self.nx[2]//2+1
        self.nnk = self.nk.copy()
        self.nnk[1] = self.nk[1]//comm.size
        self.dk = 1.0/self.L

        nny = self.nnk[1]
        iys = nny*comm.rank
        iye = iys + nny

        k2 = np.fft.rfftfreq(self.nx[2])*self.nx[2] * self.dk[2]
        k1 = np.fft.fftfreq(self.nx[1])*self.nx[1] * self.dk[1]
        k1 = k1[iys:iye].copy()
        k0 = np.fft.fftfreq(self.nx[0])*self.nx[0] * self.dk[0]

        # MPI local 3D wavemode index
        self.k = (k0, k1, k2)
        Kvec = np.array(np.meshgrid(k0, k1, k2, indexing='ij'))
        self.Kmag = np.sqrt(np.sum(np.square(Kvec), axis=0))

        return

    # -------------------------------------------------------------------------
    # Class Methods
    # -------------------------------------------------------------------------
    def flame_extents(self, Y):
        YT = self.z2y_slab_exchange(Y)

        zfs = np.min(np.nonzero(YT > 0.001)[0])
        zfe = np.max(np.nonzero(YT < 0.999)[0])

        zfs = self.comm.allreduce(zfs, op=MPI.MIN)
        zfe = self.comm.allreduce(zfe, op=MPI.MAX)

        return zfs, zfe

    def truncate_along_z(self, var, izs, ize):
        varT = self.z2y_slab_exchange(var)
        return self.y2z_slab_exchange(varT[izs:ize])

    def bincount(self, x, weights, minlength=0):
        """A very short MPI-wrapped version of `numpy.bincount` that expects
        a weights array. See `numpy.bincount` and the source code.
        """
        binsum = np.bincount(x.ravel(), weights.ravel(), minlength)

        out = None
        if self.comm.rank == 0:
            out = np.empty_like(binsum)

        self.comm.Reduce(binsum, out, op=MPI.SUM, root=0)

        if self.comm.rank !=0:
            out = 0.0

        return out

    def fft(self, var):
        """
        Convenience function for MPI-distributed 3D r2c FFT of scalar.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.rfft3(self.comm, var)

    def ifft(self, var):
        """
        Convenience function for MPI-distributed 3D c2r IFFT of scalar.
        """
        assert self._decomp == 1, "ERROR decomp not 1D"

        return tcfft.irfft3(self.comm, var)

    def vfft(self, var):
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

    def ivfft(self, fvar):
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
###############################################################################


class turbFlameArrhFilteredStats(turbFlameAnalyzer):
    """[summary]

    [description]
    """

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, nxfile, nxfft, nxstats, L=[2*np.pi]*3, N=[512]*3,
                 idir='./analysis/', odir='./analysis/', pid='test'):

        # ------------------------------------------------------------------
        # call turbFlameAnalyzer.__init__() to set up FFT variables with
        # domain size `nxfft`
        super().__init__(L=L*nxfft/N, N=nxfft, odir=odir, pid=pid)
        assert COMM.rank == self.comm.rank

        self.reader = mpiFileIO(N=nxfile, ixs=None, ixe=nxfft, idir=idir)
        assert COMM.rank == self.reader.comm.rank

        self.writer = mpiFileIO(N=nxstats, idir=idir, odir=idir)

        self._nnxf = self._nnx

        self._L = L*nxstats/N
        self._nx = nxstats
        self._dx = self._L/self._nx
        dims = self.comm.dims
        assert np.all(np.mod(self._nx, dims) == 0)
        self._nnx = self._nx//dims
        self._ixs = self._nnx*self.comm.coords
        self._ixe = self._ixs+self._nnx

    # -------------------------------------------------------------------------
    # Class Properties
    # -------------------------------------------------------------------------
    @property
    def nnxf(self):
        return self._nnxf

    # -------------------------------------------------------------------------
    # Class Methods
    # -------------------------------------------------------------------------
    def Read_all(self, var_name, tstep):
        # returns local subarray from file
        return self.reader.Read_all(f"{var_name}/{var_name}_{tstep}.bin")

    def Write_all(self, var, var_name, tstep):
        fname = f"{var_name}/{var_name}_{tstep}.bin"
        # returns MPI-IO status
        return self.writer.Write_all(fname, var)

    def dump_slices(self, var, var_name, ix=None, jy=0, kz=None):
        """Dump 2D slices from 3D scalar field.

        `dump_slices` currently assumes 1D domain decomposition along
        0th axis. Would definitely need to rethink this for a generic
        N-dimensional domain decomposition."""

        if np.isscalar(ix):
            ix = (ix, )
        if np.isscalar(jy):
            jy = (jy, )
        if np.isscalar(kz):
            kz = (kz, )
        x_slices = None
        if ix:
            data = self.comm.gather(var[:, :, ix])
            if self.comm.rank == 0:
                x_slices = np.concatenate(data, axis=0)
                filename = f'{self.odir}/{self.pid}_{var_name}-xslices.npy'
                np.save(filename, x_slices.astype('f4'), allow_pickle=False)

        y_slices = None
        if jy:
            data = self.comm.gather(var[:, jy, :])
            if self.comm.rank == 0:
                y_slices = np.concatenate(data, axis=0)
                filename = f'{self.odir}/{self.pid}_{var_name}-yslices.npy'
                np.save(filename, y_slices.astype('f4'), allow_pickle=False)

        z_slices = None
        if kz:
            krange = range(self.ixs[0], self.ixe[0])
            bool_idx = [k in np.atleast_1d(kz) for k in krange]
            data = self.comm.gather(var[bool_idx, :, :])
            if self.comm.rank == 0:
                z_slices = np.concatenate(data, axis=0)
                filename = f'{self.odir}/{self.pid}_{var_name}-zslices.npy'
                np.save(filename, z_slices.astype('f4'), allow_pickle=False)

        return x_slices, y_slices, z_slices
