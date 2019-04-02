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
import cantera as ct
import sys
import os

from teslacu import mpiAnalyzer, mpiReader, get_inputs, timeofday
from teslacu import scalar_analysis  # , vector_analysis, gradient_analysis
from teslacu.stats import psum

comm = MPI.COMM_WORLD


def autoignition_chemical_statistics(args):
    if comm.rank == 0:
        print("Python MPI job `autoignition_chemical_statistics' "
              "started with {} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint,
     _, _, _, _, _) = args

    L = L[0]  # *0.01
    N = N[0]  # N is a len(1) list, *sigh*

    nr = (ire-irs)/rint + 1
    # nt = (ite-its)/tint + 1
    Ne = nr*N**3

    prefix = ['Density',                # 0
              'Velocity1',              # 1
              'Velocity2',              # 2
              'Velocity3',              # 3
              'Temperature',            # 4
              'Bulk_Viscosity',         # 5
              'H_mass_fraction',        # 6
              'H2_mass_fraction',       # 7
              'O_mass_fraction',        # 8
              'O2_mass_fraction',       # 9
              'OH_mass_fraction',       # 10
              'H2O_mass_fraction',      # 11
              'HO2_mass_fraction',      # 12
              'H2O2_mass_fraction']     # 13

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
                       N=[N]*3, nh=None, periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer(comm=comm, odir=odir, pid=pid, L=[L]*3, N=[N]*3)
    analyzer.tol = 1.0e-16  # I don't remember what this is for!

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{0}/{0}_{1}_{2}.bin'.format

    dTstr = '\\frac{\partial T}{\partial x_i}'
    dTdTstr = '{0}{0}'.format(dTstr)
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)

    s[0] = 9
    Y = np.empty(s, dtype=np.float64)
    iH2 = 1

    gas = ct.Solution('/p/home/ctowery/python_codes/H2_san_diego.cti')

    MIN = MPI.MIN
    MAX = MPI.MAX
    SUM = MPI.SUM
    allreduce = comm.allreduce

    for it in range(its, ite+1, tint):
        tstep = str(it).zfill(4)

        emin = np.empty(50)
        emax = np.empty(50)
        emean= np.zeros(50)
        emavg= np.zeros(50)

        emin[:] = np.inf
        emax[:] = np.NINF

        for ir in range(irs, ire+1, rint):

            rho = reader.read_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.read_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.read_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.read_variable(fmt(prefix[3], tstep, ir))
            T = reader.read_variable(fmt(prefix[4], tstep, ir))
            # nub = reader.read_variable(fmt(prefix[5], tstep, ir))
            # Y[0] = reader.read_variable(fmt(prefix[6], tstep, ir))
            # Y[1] = reader.read_variable(fmt(prefix[7], tstep, ir))
            # Y[2] = reader.read_variable(fmt(prefix[8], tstep, ir))
            # Y[3] = reader.read_variable(fmt(prefix[9], tstep, ir))
            # Y[4] = reader.read_variable(fmt(prefix[10], tstep, ir))
            # Y[5] = reader.read_variable(fmt(prefix[11], tstep, ir))
            # Y[6] = reader.read_variable(fmt(prefix[12], tstep, ir))
            # Y[7] = reader.read_variable(fmt(prefix[13], tstep, ir))
            # P = reader.read_variable(fmt(prefix[5], tstep, ir))
            # Cp = reader.read_variable(fmt(prefix[6], tstep, ir))
            # nu = reader.read_variable(fmt(prefix[7], tstep, ir))
            # alpha = reader.read_variable(fmt(prefix[9], tstep, ir))
            # Dfuel = reader.read_variable(fmt(prefix[10], tstep, ir))

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables loaded into memory')

            # -----------------------------------------------------------------
            # rho *= 1000.0
            # u *= 0.01
            # nub *= 0.0001
            # P *= 0.1
            # nu *= 0.0001
            # alpha *= 0.0001
            # Dfuel *= 0.0001

            rekin = 0.5*rho*np.sum(np.square(u), axis=0)
            Smm = analyzer.div(u)
            gradT = analyzer.scl_grad(T)
            dTdT = np.sum(np.square(gradT), axis=0)

            # Y[8] = 1.0 - np.sum(Y[:-1], axis=0)
            # P = np.empty_like(T)
            # beta = np.empty_like(T)
            # HRR = np.empty_like(T)
            # FCR = np.empty_like(T)

            # for k in range(analyzer.nnx[0]):
            #     for j in range(analyzer.nnx[1]):
            #         for i in range(analyzer.nnx[2]):
            #             gas.TDY = T[k, j, i], rho[k, j, i], Y[:, k, j, i]
            #             P[k, j, i] = gas.P
            #             # HRR[k, j, i] = - np.dot(gas.net_production_rates,
            #             #                         gas.partial_molar_enthalpies)
            #             # FCR[k, j, i] = - (gas.net_production_rates[iH2]
            #             #                   *gas.molecular_weights[iH2]
            #             #                   /rho[k, j, i])
            #             # Dfuel[k, j, i] *= gas.X[iH2]/gas.Y[iH2]
            #             # beta = nub/nu

            # Sc = nu/Dfuel
            # Pr = nu/alpha
            # Le = alpha/Dfuel

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables computed')

            iv = 0

            emin[iv] = min(emin[iv], allreduce(np.min(rho), op=MIN))
            emax[iv] = max(emax[iv], allreduce(np.max(rho), op=MAX))
            emean[iv] += allreduce(psum(rho), op=SUM)
            iv+=1

            # if comm.rank % 64 == 0:
            #     print (update(timeofday(), tstep, ir, comm.rank)
            #            +'rho min max mean computed')

            # emin[iv] = min(emin[iv], allreduce(np.min(P), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(P), op=MAX))
            # emean[iv] += allreduce(psum(P), op=SUM)
            # iP = iv
            # iv+=1

            vm = np.sum(np.square(u), axis=0)
            # emin[iv] = min(emin[iv], allreduce(np.min(vm), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(vm), op=MAX))
            emean[iv] += allreduce(psum(vm), op=SUM)
            iu2 = iv
            iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(rekin), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(rekin), op=MAX))
            emean[iv] += allreduce(psum(rekin), op=SUM)
            iKE = iv
            iv+=1

            emin[iv] = min(emin[iv], allreduce(np.min(T), op=MIN))
            emax[iv] = max(emax[iv], allreduce(np.max(T), op=MAX))
            emean[iv] += allreduce(psum(T), op=SUM)
            emavg[iv] += allreduce(psum(rho*T), op=SUM)
            iT = iv
            iv+=1

            emin[iv] = min(emin[iv], allreduce(np.min(gradT), op=MIN))
            emax[iv] = max(emax[iv], allreduce(np.max(gradT), op=MAX))
            emean[iv] += allreduce(psum(gradT), op=SUM)
            emavg[iv] += allreduce(psum(rho*gradT), op=SUM)
            igT = iv
            iv+=1

            emin[iv] = min(emin[iv], allreduce(np.min(dTdT), op=MIN))
            emax[iv] = max(emax[iv], allreduce(np.max(dTdT), op=MAX))
            emean[iv] += allreduce(psum(dTdT), op=SUM)
            emavg[iv] += allreduce(psum(rho*dTdT), op=SUM)
            idT = iv
            iv+=1

            # A = analyzer.grad(u)
            # Smm = np.einsum('ii...', A)
            emin[iv] = min(emin[iv], allreduce(np.min(Smm), op=MIN))
            emax[iv] = max(emax[iv], allreduce(np.max(Smm), op=MAX))
            emean[iv] += allreduce(psum(Smm), op=SUM)
            emavg[iv] += allreduce(psum(rho*Smm), op=SUM)
            iD = iv
            iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(beta), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(beta), op=MAX))
            # emean[iv] += allreduce(psum(beta), op=SUM)
            # emavg[iv] += allreduce(psum(rho*beta), op=SUM)
            # ib = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(Sc), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(Sc), op=MAX))
            # emean[iv] += allreduce(psum(Sc), op=SUM)
            # emavg[iv] += allreduce(psum(rho*Sc), op=SUM)
            # iSc = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(Pr), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(Pr), op=MAX))
            # emean[iv] += allreduce(psum(Pr), op=SUM)
            # emavg[iv] += allreduce(psum(rho*Pr), op=SUM)
            # iPr = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(Le), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(Le), op=MAX))
            # emean[iv] += allreduce(psum(Le), op=SUM)
            # emavg[iv] += allreduce(psum(rho*Le), op=SUM)
            # iLe = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(HRR), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(HRR), op=MAX))
            # emean[iv] += allreduce(psum(HRR), op=SUM)
            # emavg[iv] += allreduce(psum(rho*HRR), op=SUM)
            # idE = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(FCR), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(FCR), op=MAX))
            # emean[iv] += allreduce(psum(FCR), op=SUM)
            # emavg[iv] += allreduce(psum(rho*FCR), op=SUM)
            # idY = iv
            # iv+=1

            # emin[iv] = min(emin[iv], allreduce(np.min(Y[iH2]), op=MIN))
            # emax[iv] = max(emax[iv], allreduce(np.max(Y[iH2]), op=MAX))
            # emean[iv] += allreduce(psum(Y[iH2]), op=SUM)
            # emavg[iv] += allreduce(psum(rho*Y[iH2]), op=SUM)
            # iY = iv
            # iv+=1

            comm.Barrier()
            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'min/max loop finished')

            # -----------------------------------------------------------------

        emean /= Ne
        rhom = emean[0]
        emavg /= rhom*Ne

        # ---------------------------------------------------------------------
        # BEGIN ANALYSIS

        for ir in range(irs, ire+1, rint):
            analyzer.mpi_moments_file = '%s%s-%s_%s.moments' % (
                                        analyzer.odir, pid, tstep, str(ir))
            analyzer.prefix = '%s-%s_%s_' % (pid, tstep, str(ir))
            analyzer.Nx = Ne
            if comm.rank == 0:
                try:
                    os.remove(analyzer.mpi_moments_file)
                except OSError:
                    pass

            rho = reader.read_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.read_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.read_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.read_variable(fmt(prefix[3], tstep, ir))
            T = reader.read_variable(fmt(prefix[4], tstep, ir))
            # nub = reader.read_variable(fmt(prefix[5], tstep, ir))
            # Y[0] = reader.read_variable(fmt(prefix[6], tstep, ir))
            # Y[1] = reader.read_variable(fmt(prefix[7], tstep, ir))
            # Y[2] = reader.read_variable(fmt(prefix[8], tstep, ir))
            # Y[3] = reader.read_variable(fmt(prefix[9], tstep, ir))
            # Y[4] = reader.read_variable(fmt(prefix[10], tstep, ir))
            # Y[5] = reader.read_variable(fmt(prefix[11], tstep, ir))
            # Y[6] = reader.read_variable(fmt(prefix[12], tstep, ir))
            # Y[7] = reader.read_variable(fmt(prefix[13], tstep, ir))
            # P = reader.read_variable(fmt(prefix[5], tstep, ir))
            # Cp = reader.read_variable(fmt(prefix[6], tstep, ir))
            # nu = reader.read_variable(fmt(prefix[7], tstep, ir))
            # alpha = reader.read_variable(fmt(prefix[9], tstep, ir))
            # Dfuel = reader.read_variable(fmt(prefix[10], tstep, ir))

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables loaded into memory')

            # -----------------------------------------------------------------
            # rho *= 1000.0
            # u *= 0.01
            # nub *= 0.0001
            # P *= 0.1
            # nu *= 0.0001
            # alpha *= 0.0001
            # Dfuel *= 0.0001

            rekin = 0.5*rho*np.sum(np.square(u), axis=0)
            Smm = analyzer.div(u)
            gradT = analyzer.scl_grad(T)
            dTdT = np.sum(np.square(gradT), axis=0)

            # Y[8] = 1.0 - np.sum(Y[:-1], axis=0)
            # P = np.empty_like(T)
            # beta = np.empty_like(T)
            # HRR = np.empty_like(T)
            # FCR = np.empty_like(T)

            # for k in range(analyzer.nnx[0]):
            #     for j in range(analyzer.nnx[1]):
            #         for i in range(analyzer.nnx[2]):
            #             gas.TDY = T[k, j, i], rho[k, j, i], Y[:, k, j, i]
            #             P[k, j, i] = gas.P
            #             # HRR[k, j, i] = np.sum(gas.net_rates_of_progress
            #             #                       *gas.delta_enthalpy)
            #             # FCR[k, j, i] = -gas.net_production_rates[iH2]
            #             # Dfuel[k, j, i] *= gas.X[iH2]/gas.Y[iH2]
            #             # beta = nub/nu

            # Sc = nu/Dfuel
            # Pr = nu/alpha
            # Le = alpha/Dfuel

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables computed')

            # -----------------------------------------------------------------

            analyzer.spectral_density(u, 'u', 'velocity PSD\t%s'
                                      % Ek_fmt('u_i'))
            # if comm.rank == 0:
            #     with open(analyzer.mpi_moments_file, 'a') as fh:
            #         fh.write('{:s}\t{:14.8e}\n'.format(
            #             'Velocity spectrum integral scale', emean[1]))

            vm = np.sum(np.square(u), axis=0)
            analyzer.write_mpi_moments(vm, 'velocity squared', None, None,
                                       emean[iu2])
            analyzer.write_mpi_moments(vm, 'm.w. velocity squared', rho, rhom,
                                       emavg[iu2])

            # scalar_analysis(analyzer, vm, (emin[iu2], emax[iu2]), emean[iu2],
            #                 None, None, 'uiui', 'velocity squared', 'u_iu_i')

            # scalar_analysis(analyzer, vm, (emin[iu2], emax[iu2]), emavg[iu2],
            #                 rho, rhom,
            #                 'uiui_tilde', 'm.w. velocity squared', 'u_iu_i')

            analyzer.write_mpi_moments(vm, 'kinetic energy', None, None,
                                       emean[iKE])
            # scalar_analysis(analyzer, rekin, (emin[iKE], emax[iKE]),
            #                 emean[iKE], None, None,
            #                 'rekin', 'kinetic energy', '\\rho u_iu_i/2')

            v = np.sqrt(rho)*u
            analyzer.spectral_density(v, 'v', 'kinetic energy PSD\t%s'
                                      % Ek_fmt('\\sqrt{{\\rho}}u_i'))
            # if comm.rank == 0:
            #     with open(analyzer.mpi_moments_file, 'a') as fh:
            #         fh.write('{:s}\t{:14.8e}\n'.format(
            #             'spectral KE integral scale', emean[iKE+1]))

            scalar_analysis(analyzer, Smm, (emin[iD], emax[iD]), emean[iD],
                            None, None, 'Smm', 'dilatation', '\Theta')
            scalar_analysis(analyzer, Smm, (emin[iD], emax[iD]), emavg[iD],
                            rho, rhom, 'Smm_tilde', 'm.w. dilatation',
                            '\Theta')

            comm.Barrier()
            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'\tvelocity analyses completed')
            # -----------------------------------------------------------------

            scalar_analysis(analyzer, rho, (emin[0], emax[0]), rhom,
                            None, None, 'rho', 'density', '\\rho')

            # scalar_analysis(analyzer, P, (emin[iP], emax[iP]), emean[iP],
            #                 None, None,
            #                 'P', 'pressure', 'P')

            scalar_analysis(analyzer, T, (emin[iT], emax[iT]), emean[iT],
                            None, None, 'T', 'temperature', 'T')
            scalar_analysis(analyzer, T, (emin[iT], emax[iT]), emavg[iT],
                            rho, rhom, 'T_tilde', 'm.w. temperature', 'T')

            scalar_analysis(analyzer, gradT, (emin[igT], emax[igT]),
                            emean[igT], None, None, 'gradT',
                            'temperature gradient', dTstr, norm=3.0)
            scalar_analysis(analyzer, gradT, (emin[igT], emax[igT]),
                            emavg[igT], np.tile(rho, (3, 1, 1, 1)), rhom,
                            'gradT_tilde', 'm.w. temperature gradient', dTstr,
                            norm=3.0)

            scalar_analysis(analyzer, dTdT, (emin[idT], emax[idT]), emean[idT],
                            None, None, 'dTdT', 'temperature gradient squared',
                            dTdTstr)
            scalar_analysis(analyzer, dTdT, (emin[idT], emax[idT]), emavg[idT],
                            rho, rhom, 'dTdT_tilde',
                            'm.w. temperature gradient squared', dTdTstr)

            # range1 = (emin[iP], emax[iP])
            # range2 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(P, T, 'P_T', 'P', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(P, T, 'P_T_tilde', 'P', 'T',
            #                         range1, range2, w=rho)

            # scalar_analysis(analyzer, Smm, (emin[iD], emax[iD]), emean[iD],
            #                 None, None, 'Smm', 'dilatation', '\Theta')
            # scalar_analysis(analyzer, Smm, (emin[iD], emax[iD]), emavg[iD],
            #                 rho, rhom, 'Smm_tilde', 'm.w. dilatation',
            #                 '\Theta')
            # range1 = (emin[iD], emax[iD])
            # analyzer.mpi_histogram2(Smm, T, 'Smm_T', '\Theta', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Smm, T, 'Smm_T_tilde', '\Theta', 'T',
            #                         range1, range2, w=rho)
            # range2 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(Smm, P, 'Smm_P', '\Theta', 'P',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Smm, P, 'Smm_P_tilde', '\Theta', 'P',
            #                         range1, range2, w=rho)

            comm.Barrier()
            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'\tthermodynamic analyses completed')
            # -----------------------------------------------------------------

            # scalar_analysis(analyzer, beta, (emin[ib], emax[ib]), emean[ib],
            #                 None, None, 'beta', 'viscosity ratio', '\\beta')
            # scalar_analysis(analyzer, beta, (emin[ib], emax[ib]), emavg[ib],
            #                 rho, rhom, 'beta_tilde', 'm.w. viscosity ratio',
            #                 '\\beta')
            # range1 = (emin[ib], emax[ib])
            # range2 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(beta, T, 'beta_T', '\\beta', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(beta, T, 'beta_T_tilde', '\\beta', 'T',
            #                         range1, range2, w=rho)

            # scalar_analysis(analyzer, Sc, (emin[iSc], emax[iSc]), emean[iSc],
            #                 None, None, 'Sc', 'Schmidt number', 'Sc')
            # scalar_analysis(analyzer, Sc, (emin[iSc], emax[iSc]), emavg[iSc],
            #                 rho, rhom, 'Sc_tilde', 'm.w. Schmidt number',
            #                 'Sc')
            # range1 = (emin[iSc], emax[iSc])
            # analyzer.mpi_histogram2(Sc, T, 'Sc_T', 'Sc', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Sc, T, 'Sc_T_tilde', 'Sc', 'T',
            #                         range1, range2, w=rho)

            # scalar_analysis(analyzer, Pr, (emin[iPr], emax[iPr]), emean[iPr],
            #                 None, None, 'Pr', 'Prandtl number', 'Pr')
            # scalar_analysis(analyzer, Pr, (emin[iPr], emax[iPr]), emavg[iPr],
            #                 rho, rhom, 'Pr_tilde', 'm.w. Prandtl number',
            #                 'Pr')
            # range1 = (emin[iPr], emax[iPr])
            # analyzer.mpi_histogram2(Pr, T, 'Pr_T', 'Pr', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Pr, T, 'Pr_T_tilde', 'Pr', 'T',
            #                         range1, range2, w=rho)

            # scalar_analysis(analyzer, Le, (emin[iLe], emax[iLe]), emean[iLe],
            #                 None, None, 'Le', 'Lewis number', 'Le')
            # scalar_analysis(analyzer, Le, (emin[iLe], emax[iLe]), emavg[iLe],
            #                 rho, rhom, 'Le_tilde', 'm.w. Lewis number',
            #                 'Le')
            # range1 = (emin[iLe], emax[iLe])
            # analyzer.mpi_histogram2(Le, T, 'Le_T', 'Le', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Le, T, 'Le_T_tilde', 'Le', 'T',
            #                         range1, range2, w=rho)

            # comm.Barrier()
            # if comm.rank % 64 == 0:
            #     print(update(timeofday(), tstep, ir, comm.rank)
            #           +'\ttransport analyses completed')
            # -----------------------------------------------------------------

            # scalar_analysis(analyzer, Y[iH2], (emin[iY], emax[iY]), emean[iY],
            #                 None, None, 'Y', 'H$_2$ mass fraction',
            #                 'Y_\mathrm{H2}')
            # scalar_analysis(analyzer, Y[iH2], (emin[iY], emax[iY]), emavg[iY],
            #                 rho, rhom, 'Y_tilde', 'm.w. H$_2$ mass fraction',
            #                 'Y_\mathrm{H2}')
            # range1 = (emin[iY], emax[iY])
            # range2 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(Y[iH2], T, 'Y_T', 'Y_\mathrm{H2}', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(Y[iH2], T, 'Y_T_tilde', 'Y_\mathrm{H2}',
            #                         'T', range1, range2, w=rho)

            # scalar_analysis(analyzer, FCR, (emin[idY], emax[idY]), emean[idY],
            #                 None, None, 'FCR', 'H$_2$ consumption rate',
            #                 '-dY_\mathrm{H2}/dt')
            # scalar_analysis(analyzer, FCR, (emin[idY], emax[idY]), emavg[idY],
            #                 rho, rhom, 'FCR_tilde',
            #                 'm.w. H$_2$ consumption rate',
            #                 '-dY_\mathrm{H2}/dt')
            # range1 = (emin[idY], emax[idY])
            # analyzer.mpi_histogram2(FCR, T, 'FCR_T', '-dY_\mathrm{H2}/dt', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(FCR, T, 'FCR_T_tilde',
            #                         '-dY_\mathrm{H2}/dt', 'T',
            #                         range1, range2, w=rho)

            # scalar_analysis(analyzer, HRR, (emin[idE], emax[idE]), emean[idE],
            #                 None, None, 'HRR', 'heat release rate',
            #                 'HRR')
            # scalar_analysis(analyzer, HRR, (emin[idE], emax[idE]), emavg[idE],
            #                 rho, rhom, 'HRR_tilde',
            #                 'm.w. heat release rate',
            #                 'HRR')
            # range1 = (emin[idE], emax[idE])
            # analyzer.mpi_histogram2(HRR, T, 'HRR_T', 'HRR', 'T',
            #                         range1, range2)
            # analyzer.mpi_histogram2(HRR, T, 'HRR_T_tilde',
            #                         'HRR', 'T',
            #                         range1, range2, w=rho)

    if comm.rank == 0:
        print("Python MPI job `autoignition_chemical_statistics'"
              " finished at "+timeofday())

    return None


if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    autoignition_chemical_statistics(get_inputs())
