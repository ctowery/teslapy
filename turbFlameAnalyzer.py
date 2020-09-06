"""
"""
from mpi4py import MPI
import numpy as np
import argparse
# from memory_profiler import profile

from .mpiAnalyzer import _hitAnalyzer
from .mpiFileIO import mpiFileIO
from . import fft as tcfft

__all__ = []

COMM = MPI.COMM_WORLD


###############################################################################
class turbFlameAnalyzer(_hitAnalyzer):
    """
    class turbFlameAnalyzer(_baseAnalyzer) ...
    Empty Docstring!
    """

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, comm=COMM, L=np.pi, N=512, odir='./', pid='test'):

        ndims = 3
        decomp = 1
        # periodic = [True, ]*3
        method = 'akima_flux_diff'

        super().__init__(comm, odir, pid, ndims, decomp, L, N, method)

        self._config = "Coarse-graining analysis of `turbflame`-like data"

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

    def dump_slices(self, var, ix=0, jy=0, kz=None, tag=None):
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

        # --------------------------------------------------------------
        x_slices = None
        if ix:
            data = self.comm.gather(var[:, :, ix])
            if self.comm.rank == 0:
                x_slices = np.concatenate(data, axis=0).astype('f4')

        # --------------------------------------------------------------
        y_slices = None
        if jy:
            data = self.comm.gather(var[:, jy, :])
            if self.comm.rank == 0:
                y_slices = np.concatenate(data, axis=0).astype('f4')

        # --------------------------------------------------------------
        z_slices = None
        if kz:
            krange = range(self.ixs[0], self.ixe[0])
            bool_idx = [k in np.atleast_1d(kz) for k in krange]
            data = self.comm.gather(var[bool_idx, :, :])
            if self.comm.rank == 0:
                z_slices = np.concatenate(data, axis=0).astype('f4')

        # --------------------------------------------------------------
        if self.comm.rank == 0 and tag is not None:
            filename = f'{self.odir}/{self.pid}_{tag}-slices.npz'
            np.savez(filename, allow_pickle=False,
                     x_slices=x_slices, y_slices=y_slices, z_slices=z_slices,
                     x_locs=ix, y_locs=jy, z_locs=kz)

        return x_slices, y_slices, z_slices

    def dump_2d_diagnostics(self, var, tag=None):
        """Dump 2D slices for diagnostic exploration from 3D scalar field.

        `dump_2d_diagnostics` currently assumes 1D domain decomposition along
        0th axis. Would definitely need to rethink this for a generic
        N-dimensional domain decomposition."""

        dx = var.shape[2]//2
        ix = (0, dx)

        dy = var.shape[1]//2
        jy = (0, dy)

        kzs = self.ixs[0]
        kze = self.ixe[0]
        kz = np.where(np.arange(kzs, kze) % 4 == 0)[0]

        # --------------------------------------------------------------
        x_slices = None
        data = self.comm.gather(var[kz, ::4, ix])
        if self.comm.rank == 0:
            x_slices = np.concatenate(data, axis=0).astype('f4')

        x_avg = None
        data = self.comm.gather(var[kz, ::4, :].mean(axis=2))
        if self.comm.rank == 0:
            x_avg = np.concatenate(data, axis=0).astype('f4')

        # --------------------------------------------------------------
        y_slices = None
        data = self.comm.gather(var[kz, jy, ::4])
        if self.comm.rank == 0:
            y_slices = np.concatenate(data, axis=0).astype('f4')

        y_avg = None
        data = self.comm.gather(var[kz, :, ::4].mean(axis=1))
        if self.comm.rank == 0:
            y_avg = np.concatenate(data, axis=0).astype('f4')

        # --------------------------------------------------------------
        if self.comm.rank == 0 and tag is not None:
            filename = f'{self.odir}/{self.pid}_{tag}-slices.npz'
            np.savez(filename, allow_pickle=False,
                     x_slices=x_slices, y_slices=y_slices,
                     x_locs=ix, y_locs=jy,
                     x_avg=x_avg, y_avg=y_avg)

        return x_avg, y_avg


###############################################################################
class turbFlameArrhFilteredStats(turbFlameAnalyzer):
    """[summary]

    [description]
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
    parser.add_argument('--iz0', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--speed', type=float)
    # parser.add_argument('--jfs', type=float)
    # parser.add_argument('--jfe', type=float)
    # parser.add_argument('--fint', type=float)

    # -------------------------------------------------------------------------
    # Class Instantiator
    # -------------------------------------------------------------------------
    def __init__(self, nxfile, nxfft, nxstats, L=[2*np.pi]*3, N=[512]*3,
                 idir='./analysis/', odir='./analysis/', pid='test'):

        # ------------------------------------------------------------------
        # call turbFlameAnalyzer.__init__() to set up FFT variables with
        # domain size `nxfft`
        super().__init__(L=L*nxfft/N, N=nxfft, odir=odir, pid=pid)
        self._config = "Coarse-graining analysis of `turbflame`-like data"

        self.reader = mpiFileIO(N=nxfile, ixs=None, ixe=nxfft, idir=idir)
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
