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
from teslacu import mpiAnalyzer, mpiReader, get_inputs, timeofday
# from teslacu import scalar_analysis  # , vector_analysis, gradient_analysis
from teslacu.stats import psum

import sys
# import os
comm = MPI.COMM_WORLD


def autoignition_chemical_slices(args):
    if comm.rank == 0:
        print("Python MPI job `autoignition_chemical_slices' "
              "started with {} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint,
     _, _, _, _, _) = args

    L = L[0]*0.01
    N = N[0]  # N is a len(1) list, *sigh*

    # nr = (ire-irs)/rint + 1
    # nt = (ite-its)/tint + 1
    # Ne = nr*N**3

    # prefix = ['Density',                # 0
    #           'Velocity1',              # 1
    #           'Velocity2',              # 2
    #           'Velocity3',              # 3
    #           'Temperature',            # 4
    #           'Bulk_Viscosity',         # 5
    #           'H_mass_fraction',        # 6
    #           'H2_mass_fraction',       # 7
    #           'O_mass_fraction',        # 8
    #           'O2_mass_fraction',       # 9
    #           'OH_mass_fraction',       # 10
    #           'H2O_mass_fraction',      # 11
    #           'HO2_mass_fraction',      # 12
    #           'H2O2_mass_fraction']     # 13

    prefix = ['Density',                # 0
              'Velocity1',              # 1
              'Velocity2',              # 2
              'Velocity3',              # 3
              'Temperature',            # 4
              'CH4_mass_fraction',      # 5
              'OH_mass_fraction',       # 6
              'CH2O_mass_fraction',     # 7
              'C2H4_mass_fraction', 'H2_mass_fraction', 'CH3_mass_fraction',
              'H_mass_fraction', 'O_mass_fraction', 'HO2_mass_fraction',
              'H2O_mass_fraction', 'H2O2_mass_fraction', 'O2_mass_fraction',
              'CH3OH_mass_fraction', 'CO_mass_fraction', 'CO2_mass_fraction',
              'C2H2_mass_fraction', 'C2H6_mass_fraction',
              'CH2CHO_mass_fraction',   # 22
              'Bulk_Viscosity',
              ]

    # 'Pressure', 'Heat_Capacity', 'Shear_Viscosity',
    # 'Thermal_Diffusivity', 'Fuel_Mass_Diffusivity',

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Job started with improper number of MPI tasks for the '
                  'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader(comm=comm, idir=idir, decomp=[True, False, False],
                       N=N, nh=None, periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer(comm=comm, odir=odir, pid=pid, L=L, N=N)

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{0}/{0}_{1}.bin'.format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    # u = np.empty(s, dtype=np.float64)

    s = list(analyzer.nnx)
    s.insert(0, 3)
    Y = np.empty(s, dtype=np.float64)

    # gas = ct.Solution('/p/home/ctowery/python_codes/H2_san_diego.cti')
    # iH2 = 1
    # iOH = 4
    # iH20= 5

    iCH4 = 5
    iOH = 6
    iFH = 7  # 'FH' = formaldehyde

    for it in range(its, ite+1, tint):
        tstep = str(it).zfill(4)

        for ir in range(irs, ire+1, rint):

            # rho = reader.read_variable(fmt(prefix[0], tstep, ir))
            # u[0] = reader.read_variable(fmt(prefix[1], tstep, ir))
            # u[1] = reader.read_variable(fmt(prefix[2], tstep, ir))
            # u[2] = reader.read_variable(fmt(prefix[3], tstep, ir))
            T = reader.read_variable(fmt(prefix[4], tstep))
            # nub = reader.read_variable(fmt(prefix[5], tstep, ir))
            Y[0] = reader.read_variable(fmt(prefix[5], tstep))
            Y[1] = reader.read_variable(fmt(prefix[6], tstep))
            Y[2] = reader.read_variable(fmt(prefix[7], tstep))
            # Y[3] = reader.read_variable(fmt(prefix[9], tstep, ir))
            # Y[4] = reader.read_variable(fmt(prefix[10], tstep, ir))
            # Y[5] = reader.read_variable(fmt(prefix[11], tstep, ir))
            # Y[6] = reader.read_variable(fmt(prefix[12], tstep, ir))
            # Y[7] = reader.read_variable(fmt(prefix[13], tstep, ir))
            # nu = reader.read_variable(fmt(prefix[7], tstep, ir))
            # alpha = reader.read_variable(fmt(prefix[9], tstep, ir))

            if comm.rank % 32 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables loaded into memory')

            # -----------------------------------------------------------------
            # rho *= 1000.0
            # u *= 0.01
            # nub *= 0.0001
            # nu *= 0.0001

            # Y[-1] = 1.0 - np.sum(Y[:-1], axis=0)
            # Smm = analyzer.div(u)
            # beta = nub/nu
            # P = np.empty_like(T)
            # HRR = np.empty_like(T)
            # FCR = np.empty_like(T)
            # dT=np.sqrt(np.sum(np.square(analyzer.scl_grad(T)), axis=0))
            # dY=np.sqrt(np.sum(np.square(analyzer.scl_grad(Y[iH2])), axis=0))
            # dP=np.sqrt(np.sum(np.square(analyzer.scl_grad(P)), axis=0))

            # for k in range(0, 1):
            #     for j in range(reader.nnx[1]):
            #         for i in range(reader.nnx[2]):
            #             gas.TDY = T[k, j, i], rho[k, j, i], Y[:, k, j, i]
            #             P[k, j, i] = gas.P
            #             HRR[k, j, i] = - np.dot(gas.net_production_rates,
            #                                     gas.partial_molar_enthalpies)
            #             FCR[k, j, i] = - (gas.net_production_rates[iOH]
            #                               *gas.molecular_weights[iOH]
            #                               /rho[k, j, i])

            if comm.rank == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables computed', flush=True)

                with open('{}{}-xy_slices_{}_{}_{}.bin'.format(
                        odir, pid, tstep, ir, comm.rank), 'w') as fh:
                    # u[0, 0].tofile(fh)
                    # u[1, 0].tofile(fh)
                    # u[2, 0].tofile(fh)
                    # rho[0].tofile(fh)
                    T[0].tofile(fh)
                    Y[0, 0].tofile(fh)
                    Y[1, 0].tofile(fh)
                    Y[2, 0].tofile(fh)

            # u0 = analyzer.z2y_slab_exchange(u[0])
            # u1 = analyzer.z2y_slab_exchange(u[1])
            # u2 = analyzer.z2y_slab_exchange(u[2])
            # rho = analyzer.z2y_slab_exchange(rho)

            T = analyzer.z2y_slab_exchange(T)
            Y0 = analyzer.z2y_slab_exchange(Y[0])
            Y1 = analyzer.z2y_slab_exchange(Y[1])
            Y2 = analyzer.z2y_slab_exchange(Y[2])

            # HRR = analyzer.z2y_slab_exchange(HRR)
            # FCR = analyzer.z2y_slab_exchange(FCR)
            # Smm = analyzer.z2y_slab_exchange(Smm)

            if comm.rank == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables transposed', flush=True)

                with open('{}{}-xz_slices_{}_{}_{}.bin'.format(
                        odir, pid, tstep, ir, comm.rank), 'w') as fh:
                    # u0[0].tofile(fh)
                    # u1[0].tofile(fh)
                    # u2[0].tofile(fh)
                    # rho[0].tofile(fh)
                    T[:, 0].tofile(fh)
                    Y0[:, 0].tofile(fh)
                    Y1[:, 0].tofile(fh)
                    Y2[:, 0].tofile(fh)

    if comm.rank == 0:
        print("Python MPI job `chemical_variable_slices'"
              " finished at "+timeofday())

    return


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    autoignition_chemical_slices(get_inputs())
