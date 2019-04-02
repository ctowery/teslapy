"""
Empty Docstring!
"""

from mpi4py import MPI

import numpy as np
from scipy.linalg import toeplitz
# from math import log, sqrt, log10

import cantera as ct
from shkdet import CV_scales, ZND_scales
from teslacu import timeofday

import argparse


###############################################################################
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


parser = argparse.ArgumentParser(prog='isosurface_profile_postprocessing',
                                 add_help=False)
parser.add_argument('-f', type=open, action=LoadInputFile, metavar='file')
parser.add_argument('--Mc', type=str)
parser.add_argument('--run', type=int)
parser.add_argument('--rho', type=float)
parser.add_argument('--eps', type=float)
parser.add_argument('--tau', type=float)
parser.add_argument('--Ti', type=float)
parser.add_argument('--Pi', type=float)
parser.add_argument('--dx_mcubes', type=int)
parser.add_argument('--dx_interp', type=float)


###############################################################################
def isosurface_profile_postprocessing():
    """
    Empty Docstring!
    """
    # ------------------------------------------------------------------
    # 0) DO SOME SETUP WORK
    # ------------------------------------------------------------------
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print('\n', flush=True)
        print("Python MPI job `isosurface_profile_postprocessing' started "
              "with {} tasks at {}.".format(comm.size, timeofday()))
        print(flush=True)

    H2_SD14 = ct.Solution('/p/home/ctowery/python_codes/H2_san_diego.cti')
    X0 = 'H2:2, O2:1, N2:3.76'

    iH2 = 1  # index for Cantera species arrays
    gas = H2_SD14

    K = 2048
    N = 1024
    config = 'H2_chem'

    args = parser.parse_known_args()[0]

    Mc = args.Mc
    run = args.run

    eps = 1.e-4*args.eps/args.rho      # erg/g  -> J/kg
    Pi = args.Pi*ct.one_atm            # Pa (J/m^3)
    Ti = args.Ti
    tau = args.tau

    dx_mcubes = args.dx_mcubes
    dx_interp = args.dx_interp
    npmc = int(dx_mcubes/dx_interp)  # number of interp points in mcubes cell

    case = 'M{0}_K{1}_N{2}_{3}'.format(Mc, K, N, config)
    prob_id = '%s_r%d' % (case, run)
    root = '/p/work1/ctowery/autoignition/'
    idir = '%s/data/%s' % (root, case)
    odir = '%s/eulerian_analysis/%s' % (root, case)

    gas.TPX = Ti, Pi, X0
    Yinit = gas.Y[iH2]
    # vi = 1.0/gas.density

    tau_ign, tau_hm1, tau_hm2 = CV_scales(Ti, Pi, eps, tau, X0, gas)[:3]
    tau_exo = tau_hm2 - tau_hm1

    gas.TPX = Ti, Pi, X0
    reactor = ct.IdealGasReactor(gas)
    env = ct.Reservoir(ct.Solution('air.xml'))
    ct.Wall(reactor, env, A=1.0, Q=-eps*reactor.mass)
    sim = ct.ReactorNet([reactor])

    sim.advance(tau_hm1)
    Yhm1 = reactor.Y[iH2]
    # Thm1 = reactor.thermo.T
    # Phm1 = reactor.thermo.P

    sim.advance(tau_ign)
    Y_crit = Yign = reactor.Y[iH2]
    # Tign = reactor.thermo.T
    # Pign = reactor.thermo.P

    sim.advance(tau_hm2)
    Yhm2 = reactor.Y[iH2]
    # Thm2 = reactor.thermo.T
    # Phm2 = reactor.thermo.P

    sim.advance(2*tau)
    Yf = reactor.Y[iH2]
    # Tf = reactor.thermo.T
    Pf = reactor.thermo.P

    ZND_state, CJ_state = ZND_scales(Ti, Pi, X0, gas, t_scale=10*tau_exo)
    gas.state = ZND_state
    Pznd = gas.P
    gas.state = CJ_state
    P_cj = gas.P

    comm.Barrier()
    if comm.rank == 0:
        print('%s: setup complete' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 1) READ DATA FROM DISK (INTERNAL CELLS ONLY)
    # ------------------------------------------------------------------
    nfiles = 256
    nfiles_per_task = nfiles//comm.size
    profiles = []
    irs = comm.rank*nfiles_per_task
    ire = irs + nfiles_per_task
    for rank in range(irs, ire):
        save_file = '%s/%s-profiles_%03d.npz' % (odir, prob_id, rank)
        data = np.load(save_file)
        profiles.append(data['profiles'])

    profiles = np.concatenate(profiles, axis=1)
    states = profiles[3:11]  # cutting off velocity and Tgrad just for now
    nvar, nfaces, npoints = states.shape
    length = npoints//2

    comm.Barrier()
    if comm.rank == 0:
        print('%s: states loaded' % timeofday(), flush=True)

    # ---------------- JPDF(P, 1/rho) ----------------
    var1 = np.power(states[0].ravel(), -1.0)
    var2 = states[1].ravel()/ct.one_atm

    hist, xy_range = histogram2(var1, var2, comm=comm)

    if comm.rank == 0:
        # hist *= 1.0/hist.sum()  # makes this a PMF
        save_file = '%s/%s-domain-Pv_jpdf.npz' % (odir, prob_id)
        np.savez(save_file, hist=hist, xy_range=xy_range, bins=100)

    # ------------------------------------------------------------------
    # 7) CROP THE PROFILES TO JUST THE CENTERED REACTION FRONTS
    # ------------------------------------------------------------------
    # C is progress variable, 0 = unburned, 1 = burned
    states[3] = 1.0 - (states[3] - Yf)/(Yinit - Yf)
    Chm1 = 1.0 - (Yhm1 - Yf)/(Yinit - Yf)
    Cign = 1.0 - (Yign - Yf)/(Yinit - Yf)
    Chm2 = 1.0 - (Yhm2 - Yf)/(Yinit - Yf)
    C_crit = 1.0 - (Y_crit - Yf)/(Yinit - Yf)

    r = np.zeros(npoints, dtype=np.float32)
    r[:3] = 1.0
    G = toeplitz(r)
    G /= G.sum(axis=0)

    accepted = 0
    accept = np.zeros(nfaces, dtype=np.bool)
    mask = np.zeros((nfaces, npoints), dtype=np.bool)
    rejected1 = 0
    rejected2 = 0
    rejected3 = 0
    nl = 0
    nr = 0
    for f in range(nfaces):
        C = np.matmul(states[3, f], G)
        hrr = np.matmul(states[6, f], G)  # actually thermicity, not HRR
        hrr = np.matmul(hrr, G)  # double-smooth this infernal quantity!

        # -------------------------------------------
        # Step 1) find the centermost smoothed isosurface crossing and
        #         truncate profile at secondary isosurface crossings
        diff = np.diff(1 - 2*np.signbit(C - C_crit))
        smooth_crossings = np.where(diff < 0)[0] + 1

        if smooth_crossings.size == 0:  # Step 1 reject
            mask[f] = True
            rejected1 += 1
            continue

        idx = np.abs(smooth_crossings - length).argmin()
        ipc = smooth_crossings[idx]  # index of centermost negative crossing

        diff = np.diff(np.signbit(states[3, f] - C_crit))
        interp_crossings = np.where(diff)[0] + 1  # includes both directions

        for p in interp_crossings:
            if p < ipc - npmc - 1:
                mask[f, :p] = True
            elif p > ipc + npmc + 1:
                mask[f, p:] = True

        idx = np.where(~mask[f])[0]
        if idx.size <= 2:
            mask[f] = True
            rejected1 += 1
            continue

        # -------------------------------------------
        # Step 2) only keep profiles that have both reactants and products
        if C[idx].max() < Chm2 or C[idx].min() > Chm1:  # Step 2 reject
            mask[f] = True
            rejected2 += 1
            continue

        # -------------------------------------------
        # Step 3) find the centermost smoothed peak thermicity and
        #         truncate profile at secondary thermicity minima

        ips, ipe = idx[0], idx[-1]+1
        diff = np.diff(1 - 2*np.signbit(np.diff(hrr[ips:ipe])))
        local_maxima = np.where(diff < 0)[0] + ips + 1

        if local_maxima.size == 0:
            mask[f] = True
            rejected3 += 1
            continue

        ign = np.abs(local_maxima - ipc).argmin()
        ign = local_maxima[ign]  # index of centermost peak thermicity

        diff = np.diff(np.signbit(np.diff(hrr[ips:ipe])))
        extrema = np.where(diff)[0] + ips + 1

        for p in extrema:
            if p < min(ign, ipc) - 1:
                mask[f, :p] = True
            elif p > max(ign, ipc) + 1:
                mask[f, p:] = True

        if idx.size > 2:
            accepted += 1
            accept[f] = True
            ips, ipe = idx[0], idx[-1]+1

            # jps = C[ips:ipc].argmax() + ips
            # jpe = C[ipc:ipe].argmin() + ipc+1
            # mask[f, :jps] = True
            # mask[f, jpe:] = True

            nl = max(nl, ipc - ips)
            nr = max(nr, ipe - ipc)
        else:
            mask[f] = True
            rejected3 += 1

    nl = comm.allreduce(nl, op=MPI.MAX)
    nr = comm.allreduce(nr, op=MPI.MAX)

    if comm.rank == 0:
        print('%s: nl, nr = %d, %d' % (timeofday(), nl, nr), flush=True)

    # ------------------------------------------------
    # Step 5) smooth, center, and copy into to states2

    fidx = np.where(accept)[0]
    nfaces = fidx.size
    npoints = nl + nr
    states2 = np.empty((nvar, nfaces, npoints), dtype=states.dtype)
    states2.fill(np.NaN)

    for i, f in enumerate(fidx):
        idx = np.where(~mask[f])[0]
        ips, ipe = idx[0], idx[-1]+1

        profile = states[:, f, :]
        C = np.matmul(profile[3], G)

        diff = np.diff(1 - 2*np.signbit(C - C_crit))
        smooth_crossings = np.where(diff < 0)[0] + 1
        idx = np.abs(smooth_crossings - length).argmin()
        ipc = smooth_crossings[idx]

        npl = ipc - ips
        if npl > nl:
            ips += npl - nl
            npl = nl
        npr = ipe - ipc
        if npr > nr:
            ipe -= npr - nr
            npr = nr

        states2[:, i, nl-npl:nl+npr] = profile[:, ips:ipe]

    '''
    diff = np.diff(1 - 2*np.signbit(np.diff(hrr)))
    local_maxima = np.where(diff < 0)[0] + 1

    if local_maxima.size == 0:
        mask[f] = True
        rejected1 += 1
        continue

    idx = np.abs(local_maxima - length).argmin()
    ign = local_maxima[idx]  # index of centermost peak thermicity

    diff = np.diff(np.signbit(np.diff(hrr)))
    extrema = np.where(diff)[0] + 1

    for p in extrema:
        if p < ign-npmc-1:
            mask[f, :p] = True
        elif p > ign+npmc+1:
            mask[f, p:] = True

    idx = np.where(~mask[f])[0]
    ips, ipe = idx[0], idx[-1]+1
    halfmax = 0.5*hrr[ign]

    has_halfmax_pulse = (idx.size > 2 and
                         np.min(hrr[ips:ign], initial=np.inf) < halfmax and
                         np.min(hrr[ign:ipe], initial=np.inf) < halfmax)
    if has_halfmax_pulse:
        accepted += 1
        accept[f] = True

        nl = max(nl, ign - ips)
        nr = max(nr, ipe - ign)

    else:  # Step 3 reject
        mask[f] = True
        rejected3 += 1
    '''
    # -------------------------------------------
    states = None
    profiles = None
    profile = None
    mask = None

    comm.Barrier()
    if comm.rank == 0:
        print('%s: states compressed into states2' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 8) CATEGORIZE THE REACTION WAVES
    # ------------------------------------------------------------------
    cats = [[] for _ in range(12)]
    # cats[0]  = quasi-constant-pressure autoignition
    # cats[1]  = general low-pressure autoignition
    # cats[2]  = quasi-constant-volume autoignition
    # cats[3]  = possible detonation zone 1  (weakest)
    # cats[4]  =             ...                 v
    # cats[5]  =             ...                 v
    # cats[6]  =             ...                 v
    # cats[7]  =             ...                 v
    # cats[8]  =             ...                 v
    # cats[9]  =             ...                 v
    # cats[10] =             ...                 v
    # cats[11] = possible detonation zone 9  (strongest)

    for f in range(nfaces):
        idx = np.where(np.isfinite(states2[0, f]))[0]
        ips, ipe = idx[0], idx[-1]+1

        rho = states2[0, f, ips:ipe]
        P = states2[1, f, ips:ipe]
        hrr = states2[5, f, ips:ipe]

        ign = hrr.argmax()
        halfmax = 0.5*hrr[ign]
        crossings = np.where(np.diff(np.signbit(hrr - halfmax)))[0] + 1
        if crossings.size == 0:
            fhm = hrr.size - 1
            shm = 0
        elif crossings.size == 1:
            if crossings[0] < ign:
                shm = crossings[0]
                fhm = hrr.size - 1
            else:
                fhm = crossings[0] - 1
                shm = 0
        else:
            shm = crossings[0]
            fhm = crossings[-1] - 1

        Pmax = P.max()
        ipm = P.argmax()
        vshm = 1/rho[shm]
        vfhm = 1/rho[fhm]
        slope = abs((vfhm/P[fhm])*(P[shm] - P[fhm])/(vshm - vfhm))

        # --------------------------------------------- # former designation:
        if Pmax >= Pznd:
            if ipm > fhm:
                cats[11].append(f)                     # strong detonation
            elif (fhm >= ipm) and (ipm > shm):
                cats[10].append(f)                     # strong detonation
            else:
                cats[9].append(f)                      # possible zone 1
        elif (Pznd > Pmax) and (Pmax >= P_cj):
            if ipm > fhm:
                cats[8].append(f)                      # strong detonation
            elif (fhm >= ipm) and (ipm > shm):
                cats[7].append(f)                      # possible zone 2
            else:
                cats[6].append(f)                      # possible zone 3
        elif (P_cj > Pmax) and (Pmax >= 1.1*Pf):
            if ipm > fhm:
                cats[5].append(f)                      # weak detonation
            elif (fhm >= ipm) and (ipm > shm):
                cats[4].append(f)                      # possible zone 4
            else:
                cats[3].append(f)                      # pre-shocked
        else:
            if slope > 4.0:
                cats[2].append(f)                      # QCV (same)
            elif 4.0 >= slope and slope >= 0.25:
                cats[1].append(f)                      # other (same)
            else:
                cats[0].append(f)                      # QCP (same)

    if comm.rank == 0:
        print('%s: profiles categorized' % timeofday(), flush=True)

    subtot = comm.allreduce(accepted, op=MPI.SUM)
    nbad1 = comm.allreduce(rejected1, op=MPI.SUM)
    nbad2 = comm.allreduce(rejected2, op=MPI.SUM)
    nbad3 = comm.allreduce(rejected3, op=MPI.SUM)

    ncl = np.array([len(cat) for cat in cats])
    ncg = np.zeros_like(ncl)
    comm.Allreduce(ncl, ncg, op=MPI.SUM)

    if comm.rank == 0:

        total = subtot + nbad1 + nbad2 + nbad3
        print('\n############################################################')
        print('total # of profiles:                     %8d / 100.0000\n'
              % total)
        total *= 0.01
        print('accpeted profiles:                       %8d / %8.4f'
              % (subtot, subtot/total))
        print('Step 1 rejects:                          %8d / %8.4f'
              % (nbad1, nbad1/total))
        print('Step 2 rejects:                          %8d / %8.4f'
              % (nbad2, nbad2/total))
        print('Step 3 rejects:                          %8d / %8.4f\n'
              % (nbad3, nbad3/total))
        subtot *= 0.01
        print('quasi-const-pressure:                    %8d / %8.4f'
              % (ncg[0], ncg[0]/subtot))
        print('general autoignition:                    %8d / %8.4f'
              % (ncg[1], ncg[1]/subtot))
        print('quasi-const-volume:                      %8d / %8.4f\n'
              % (ncg[2], ncg[2]/subtot))
        print('detonation zone 1 (Pcj > Pmax >= Pf):    %8d / %8.4f'
              % (ncg[3], ncg[3]/subtot))
        print('detonation zone 2:                       %8d / %8.4f'
              % (ncg[4], ncg[4]/subtot))
        print('detonation zone 3:                       %8d / %8.4f\n'
              % (ncg[5], ncg[5]/subtot))
        print('detonation zone 4 (Pznd > Pmax >= Pcj):  %8d / %8.4f'
              % (ncg[6], ncg[6]/subtot))
        print('detonation zone 5:                       %8d / %8.4f'
              % (ncg[7], ncg[7]/subtot))
        print('detonation zone 6:                       %8d / %8.4f\n'
              % (ncg[8], ncg[8]/subtot))
        print('detonation zone 7 (Pmax >= Pznd):        %8d / %8.4f'
              % (ncg[9], ncg[9]/subtot))
        print('detonation zone 8:                       %8d / %8.4f'
              % (ncg[10], ncg[10]/subtot))
        print('detonation zone 9:                       %8d / %8.4f\n'
              % (ncg[11], ncg[11]/subtot))
        print('############################################################\n',
              flush=True)

    # ------------------------------------------------------------------
    # 9) ANALYZE EACH REACTION WAVE CATEGORY
    # ------------------------------------------------------------------
    names = ['quasi_CPs', 'otherigns', 'quasi_CVs',
             'det_zone1', 'det_zone2', 'det_zone3',
             'det_zone4', 'det_zone5', 'det_zone6',
             'det_zone7', 'det_zone8', 'det_zone9',
             'all']
    avg_profiles = [[] for _ in range(12)]
    avg_dict = {}
    exmpl_dict = {}

    prefix = '%s/%s-%s' % (odir, prob_id, 'all')
    all_the_histograms(comm, states2, nl, nr, prefix)

    idx = np.where(ncg > 0)[0]
    for m in idx:
        states = np.ascontiguousarray(states2[:, cats[m], :])
        prefix = '%s/%s-%s' % (odir, prob_id, names[m])
        all_the_histograms(comm, states, nl, nr, prefix)

        # no. valid profiles per stencil point
        ncpp = np.sum(np.isfinite(states[0]), axis=0)
        comm.Allreduce(MPI.IN_PLACE, ncpp, op=MPI.SUM)
        ncpp = np.fmax(1, ncpp).reshape(1, npoints)

        # average profile, including stencil points with NaNs
        avg_profiles[m] = np.nansum(states[:2], axis=1)/ncpp
        if comm.rank == 0:
            comm.Reduce(MPI.IN_PLACE, avg_profiles[m], op=MPI.SUM)
            avg_dict[names[m]] = avg_profiles[m]
        else:
            comm.Reduce(avg_profiles[m], None, op=MPI.SUM)

        if ncl[m] > 10:
            draw = np.random.choice(ncl[m], size=10, replace=False)
            examples = np.ascontiguousarray(states[:2, draw])
        else:
            examples = np.ascontiguousarray(states[:2])

        examples = comm.gather(examples, root=0)
        if comm.rank == 0:
            examples = np.concatenate(examples, axis=1)
            count = examples.shape[1]
            size = min(count, 10)
            draw = np.random.choice(count, size=size, replace=False)
            exmpl_dict[names[m]] = examples[:, draw]

    idx = np.where(ncg == 0)[0]
    for m in idx:
        avg_dict[names[m]] = np.zeros((2, 1))
        exmpl_dict[names[m]] = np.zeros((2, 10, 1))

    if comm.rank == 0:
        save_file = '%s/%s-mean_profiles.npz' % (odir, prob_id)
        np.savez(save_file, **avg_dict)

        save_file = '%s/%s-exmpl_profiles.npz' % (odir, prob_id)
        np.savez(save_file, **exmpl_dict)

        print('THE END!!!', flush=True)

    # ------------------------------------------------------------------
    # THE END!!!!!!
    # ------------------------------------------------------------------
    return


###############################################################################
def histogram(var, w=None, range=None, bins=100, comm=MPI.COMM_WORLD):
    '''
    this MPI-distributed histogram function is safe for null-sized arrays
    '''

    if range is None:
        fill = np.ma.minimum_fill_value(var)
        gmin = comm.allreduce(np.min(var, initial=fill), op=MPI.MIN)
        gmax = comm.allreduce(np.max(var, initial=-fill), op=MPI.MAX)
        range = (gmin, gmax)

    temp = np.histogram(var, bins=bins, range=range, weights=w)[0]
    hist = np.ascontiguousarray(temp)

    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, hist, op=MPI.SUM)
    else:
        comm.Reduce(hist, None, op=MPI.SUM)

    return hist, range


def histogram2(var1, var2, w=None, x_range=None, y_range=None,
               xbins=100, ybins=100, comm=MPI.COMM_WORLD):
    '''
    this MPI-distributed 2D histogram function is safe for null-sized arrays
    '''

    if x_range is None:
        fill = np.ma.minimum_fill_value(var1)
        gmin = comm.allreduce(np.min(var1, initial=fill), op=MPI.MIN)
        gmax = comm.allreduce(np.max(var1, initial=-fill), op=MPI.MAX)
        x_range = (gmin, gmax)

    if y_range is None:
        fill = np.ma.minimum_fill_value(var2)
        gmin = comm.allreduce(np.min(var2, initial=fill), op=MPI.MIN)
        gmax = comm.allreduce(np.max(var2, initial=-fill), op=MPI.MAX)
        y_range = (gmin, gmax)

    xy_range = (x_range, y_range)

    temp = np.histogram2d(var1, var2, bins=[xbins, ybins],
                          range=xy_range, weights=w)[0]
    hist = np.ascontiguousarray(temp)

    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, hist, op=MPI.SUM)
    else:
        comm.Reduce(hist, None, op=MPI.SUM)

    return hist, xy_range


def all_the_histograms(comm, states, nl, nr, prefix):
    # ------------------------------------------------------------------
    # PROFILE STENCIL CHECKS
    # ------------------------------------------------------------------
    data = states[:, np.isfinite(states[3])]        # data is a 2D array
    mstates = np.ma.array(states, mask=np.isnan(states))  # doesn't copy
    npoints = nl+nr
    x_edges = (-nl-0.5, nr-0.5)

    # ---------------- CPDF(C; stencil pt) --------------------
    cpdf = np.zeros((npoints, 100), dtype=np.int64)
    Cmin = comm.allreduce(np.min(data[3], initial=np.inf), op=MPI.MIN)
    Cmax = comm.allreduce(np.max(data[3], initial=-np.inf), op=MPI.MAX)
    for p in range(npoints):
        C = states[3, :, p]
        cpdf[p] = histogram(C, range=(Cmin, Cmax), bins=100, comm=comm)[0]

    if comm.rank == 0:
        save_file = '%s-Cx_cpdf.npz' % prefix
        np.savez(save_file, cond_pdf=cpdf,
                 extents=x_edges+(Cmin, Cmax))

    # ---------------- CPDF(hs; stencil pt) --------------------
    cpdf.fill(0.0)
    Hmin = comm.allreduce(np.min(data[4], initial=np.inf), op=MPI.MIN)
    Hmax = comm.allreduce(np.max(data[4], initial=-np.inf), op=MPI.MAX)
    for p in range(npoints):
        var = states[4, :, p]
        cpdf[p] = histogram(var, range=(Hmin, Hmax), bins=100, comm=comm)[0]

    imin = mstates[4].argmin(axis=-1) - float(nl)  # 1D size nfaces
    imax = mstates[4].argmax(axis=-1) - float(nl)  # 1D size nfaces

    hist1 = histogram(imin, range=x_edges, bins=npoints, comm=comm)[0]
    hist2 = histogram(imax, range=x_edges, bins=npoints, comm=comm)[0]

    if comm.rank == 0:
        save_file = '%s-hs_pdfs.npz' % prefix
        np.savez(save_file, cond_pdf=cpdf, min_pdf=hist1, max_pdf=hist2,
                 extents=x_edges+(Hmin, Hmax), range=x_edges)

    # ---------------- CPDF(HRR; stencil pt) --------------------
    cpdf.fill(0.0)
    Hmin = comm.allreduce(np.min(data[5], initial=np.inf), op=MPI.MIN)
    Hmax = comm.allreduce(np.max(data[5], initial=-np.inf), op=MPI.MAX)
    for p in range(npoints):
        var = states[5, :, p]
        cpdf[p] = histogram(var, range=(Hmin, Hmax), bins=100, comm=comm)[0]

    imin = mstates[5].argmin(axis=-1) - float(nl)  # 1D size nfaces
    imax = mstates[5].argmax(axis=-1) - float(nl)  # 1D size nfaces

    hist1 = histogram(imin, range=x_edges, bins=npoints, comm=comm)[0]
    hist2 = histogram(imax, range=x_edges, bins=npoints, comm=comm)[0]

    if comm.rank == 0:
        save_file = '%s-HRR_pdfs.npz' % prefix
        np.savez(save_file, cond_pdf=cpdf, min_pdf=hist1, max_pdf=hist2,
                 extents=x_edges+(Hmin, Hmax), range=x_edges)

    # ---------------- CPDF(thermicity; stencil pt) --------------------
    cpdf.fill(0.0)
    Hmin = comm.allreduce(np.min(data[6], initial=np.inf), op=MPI.MIN)
    Hmax = comm.allreduce(np.max(data[6], initial=-np.inf), op=MPI.MAX)
    for p in range(npoints):
        var = states[6, :, p]
        cpdf[p] = histogram(var, range=(Hmin, Hmax), bins=100, comm=comm)[0]

    imin = mstates[6].argmin(axis=-1) - float(nl)  # 1D size nfaces
    imax = mstates[6].argmax(axis=-1) - float(nl)  # 1D size nfaces

    hist1 = histogram(imin, range=x_edges, bins=npoints, comm=comm)[0]
    hist2 = histogram(imax, range=x_edges, bins=npoints, comm=comm)[0]

    if comm.rank == 0:
        save_file = '%s-therm_pdfs.npz' % prefix
        np.savez(save_file, cond_pdf=cpdf, min_pdf=hist1, max_pdf=hist2,
                 extents=x_edges+(Hmin, Hmax), range=x_edges)

    '''
    # ---------------- CPDF(dilatation; stencil pt) --------------------
    cpdf.fill(0.0)
    Hmin = comm.allreduce(np.min(data[7], initial=np.inf), op=MPI.MIN)
    Hmax = comm.allreduce(np.max(data[7], initial=-np.inf), op=MPI.MAX)
    for p in range(npoints):
        var = states[7, :, p]
        cpdf[p] = histogram(var, range=(Hmin, Hmax), bins=100, comm=comm)[0]

    imin = mstates[7].argmin(axis=-1) - float(nl)  # 1D size nfaces
    imax = mstates[7].argmax(axis=-1) - float(nl)  # 1D size nfaces

    hist1 = histogram(imin, range=x_edges, bins=npoints, comm=comm)[0]
    hist2 = histogram(imax, range=x_edges, bins=npoints, comm=comm)[0]

    if comm.rank == 0:
        save_file = '%s-dil_pdfs.npz' % prefix
        np.savez(save_file, cond_pdf=cpdf, min_pdf=hist1, max_pdf=hist2,
                 extents=x_edges+(Hmin, Hmax), range=x_edges)
    '''

    # ------------------------------------------------------------------
    # PROFILE DYNAMICS
    # ------------------------------------------------------------------

    # ---------------- JPDF(P, 1/rho) ----------------
    var1 = np.power(data[0], -1.0)
    var2 = data[1]/ct.one_atm

    hist, xy_range = histogram2(var1, var2, comm=comm)

    if comm.rank == 0:
        # hist *= 1.0/hist.sum()  # makes this a PMF
        save_file = '%s-Pv_jpdf.npz' % prefix
        np.savez(save_file, hist=hist, xy_range=xy_range, bins=100)

    return


###############################################################################
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    isosurface_profile_postprocessing()
