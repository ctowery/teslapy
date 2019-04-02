"""
Analysis of compressible, ideal gas, HIT using draft versions of TESLaCU
python modules.

Command Line Options:
---------------------
-i <input directory>    default: 'data/'
-o <output directory>   default: 'analysis/'
-p <problem ID>         defualt: 'no_problem_id'
-N <Nx>                 default: 512
-g <gamma>              default: 1.4
-L <L>                  default: 1.0
-R <R>                  default: 8.3144598e7/21
-r <irs:ire:rint>       default: 1:20:1
-t <its:ite:tint>       default: 1:20:1
--Texp <texp>           default: 0.7
--Tcoef <tcoef>         default: 3.1e-6
--Tmp0 <tmp0>           default: 293.0

Notes:
------

Definitions:
------------

Authors:
--------
Colin Towery, colin.towery@colorado.edu

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""
from mpi4py import MPI
import numpy as np
# import cantera as ct
import mpiAnalyzer
import mpiReader
from single_comm_functions import get_inputs, timeofday

import sys
import os
comm = MPI.COMM_WORLD


###############################################################################
def turbflame_chemical_slices(args):
    if comm.rank == 0:
        print("Python MPI job `turbflame_chemical_slices' "
              "started with {} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint,
     _, _, _, _, _) = args

    if comm.rank == 0:
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

    status = comm.bcast(status)
    if status != 0:
        MPI.Finalize()
        sys.exit(999)

    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3', 'Temperature',
              'Pressure', 'NC12H26_mass_fraction']

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots

    if N[0] % comm.size > 0:
        if comm.rank == 0:
            print('Job started with improper number of MPI tasks for the '
                  'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader.mpiBinaryReader(
                mpi_comm=comm, idir=idir, ndims=3,
                decomp=[True, False, False], nx=N, nh=None,
                periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer.factory(comm=comm, idir=idir, odir=odir,
                                   probID=pid, L=L, nx=N)

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{0}/{0}_{1}.bin'.format

    s = list(reader.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)

    # gas = ct.Solution('/p/home/ctowery/cantera/H2_san_diego.cti')

    for it in range(its, ite+1, tint):
        tstep = str(it).zfill(4)

        for ir in range(irs, ire+1, rint):

            # rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            T = reader.get_variable(fmt(prefix[4], tstep))
            P = reader.get_variable(fmt(prefix[5], tstep))
            Yf = reader.get_variable(fmt(prefix[6], tstep))

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables loaded into memory')

            # -----------------------------------------------------------------

            Enst = 0.5*np.sum(np.square(analyzer.curl(u)), axis=0)

            Enst = analyzer.z2y_slab_exchange(Enst)
            T = analyzer.z2y_slab_exchange(T)
            P = analyzer.z2y_slab_exchange(P)
            Yf = analyzer.z2y_slab_exchange(Yf)

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables computed and transposed')

            if comm.rank == comm.size//2:
                with open('{}/slices_{}.bin'.format(
                        odir, tstep), 'w') as fh:
                    Enst[:, 0, :].tofile(fh)
                    T[:, 0, :].tofile(fh)
                    P[:, 0, :].tofile(fh)
                    Yf[:, 0, :].tofile(fh)

    if comm.rank == 0:
        print("Python MPI job `turbflame_chemical_slices'"
              " finished at "+timeofday())

    return


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    turbflame_chemical_slices(get_inputs())
