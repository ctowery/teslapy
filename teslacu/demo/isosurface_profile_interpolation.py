"""
Empty Docstring!
"""

from mpi4py import MPI
import numpy as np
from numba import jit, prange, stencil
import argparse
import os

from skimage.measure import marching_cubes_lewiner as marching_cubes
import cantera as ct
from shkdet import CV_scales
from teslacu import mpiReader, timeofday


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


parser = argparse.ArgumentParser(prog='isosurface_profile_interpolation',
                                 add_help=False)
parser.add_argument('-f', type=open, action=LoadInputFile, metavar='file')
parser.add_argument('--Mc', type=str)
parser.add_argument('--run', type=int)
parser.add_argument('--tstep', type=str)
parser.add_argument('--L', type=float)
parser.add_argument('--rho', type=float)
parser.add_argument('--eps', type=float)
parser.add_argument('--tau', type=float)
parser.add_argument('--Ti', type=float)
parser.add_argument('--Pi', type=float)
parser.add_argument('--skip', type=int)
parser.add_argument('--ng_prof', type=int)
parser.add_argument('--dx_mcubes', type=int)
parser.add_argument('--dx_interp', type=float)

###############################################################################
H2_SD14 = ct.Solution('/p/home/ctowery/python_codes/H2_san_diego.cti')
X0 = 'H2:2, O2:1, N2:3.76'

nsp = 9
iH2 = 1  # index for Cantera species arrays
gas = H2_SD14

prefix = ['Velocity1',              # 0
          'Velocity2',              # 1
          'Velocity3',              # 2
          'Density',                # 3
          'Temperature',            # 4
          'H_mass_fraction',        # 5,  -9,  0
          'H2_mass_fraction',       # 6,  -8,  1
          'O_mass_fraction',        # 7,  -7,  2
          'O2_mass_fraction',       # 8,  -6,  3
          'OH_mass_fraction',       # 9,  -5,  4
          'H2O_mass_fraction',      # 10, -4,  5
          'HO2_mass_fraction',      # 11, -3,  6
          'H2O2_mass_fraction']     # 12, -2,  7

nvars = 14


###############################################################################
def isosurface_profile_interpolation():
    """
    Empty Docstring!
    """
    if MPI.COMM_WORLD.rank == 0:
        print('\n', flush=True)
        print("Python MPI job `isosurface_profile_interpolation' started with "
              "{} tasks at {}.".format(MPI.COMM_WORLD.size, timeofday()))
        print(flush=True)

    # ------------------------------------------------------------------
    # 0) DO SOME SETUP WORK
    # ------------------------------------------------------------------
    args = parser.parse_known_args()[0]

    K = 2048
    N = 1024
    config = 'H2_chem'
    Mc = args.Mc
    run = args.run
    tstep = args.tstep

    tau = args.tau
    eps = 1.e-4*args.eps/args.rho  # erg/g  -> J/kg
    Pi = args.Pi*ct.one_atm
    Ti = args.Ti
    L = 1.e-2*args.L
    dx = L/N

    case = 'M{0}_K{1}_N{2}_{3}'.format(Mc, K, N, config)
    prob_id = '%s_r%d' % (case, run)
    root = '/p/work1/ctowery/autoignition/'
    idir = '%s/data/%s' % (root, case)
    odir = '%s/eulerian_analysis/2019_03_13/%s' % (root, case)
    fmt = ('{0}/{0}_{1}_%d.bin' % run).format

    if MPI.COMM_WORLD.rank == 0:
            try:
                os.makedirs(odir)
            except OSError as e:
                if not os.path.isdir(odir):
                    raise e

    ng_akima = 3            # no. ghost cells needed for akima interpolation
    ng_prof = args.ng_prof  # no. ghost cells needed for interpolated profiles
    dx_mcubes = args.dx_mcubes
    dx_interp = args.dx_interp
    skip = args.skip

    lx = ng_prof - dx_mcubes - ng_akima
    nx = 4*int(np.log2(lx) - np.log2(dx_interp)) + 1
    npoints = 2*nx + 1
    interp_points = np.hstack((np.geomspace(-lx, -dx_interp, nx), 0,
                               np.geomspace(dx_interp, lx, nx))
                              ).reshape(npoints, 1)

    tau_ign = CV_scales(Ti, Pi, eps, tau, X0, gas)[0]
    gas.TPX = Ti, Pi, X0
    reactor = ct.IdealGasReactor(gas)
    env = ct.Reservoir(ct.Solution('air.xml'))
    ct.Wall(reactor, env, A=1.0, Q=-eps*reactor.mass)
    sim = ct.ReactorNet([reactor])
    sim.advance(tau_ign)
    iso_val = reactor.Y[iH2]

    # ------------------------------------------------------------------
    # 1) READ DATA FROM DISK (INTERNAL CELLS ONLY)
    # ------------------------------------------------------------------
    reader = mpiReader(MPI.COMM_WORLD, idir=idir, decomp=3, N=[N]*3, ndims=3,
                       periodic=[True]*3)
    comm = reader.comm
    nnx = reader.nnx
    ixe = nnx + ng_prof

    U_shape = list(nnx + 2*ng_prof)
    U_shape.insert(0, nvars)
    U = np.empty(U_shape, dtype=np.float32)

    # species mass fraction array does not have any ghost cells
    Y_shape = list(nnx)
    Y_shape.insert(0, nsp)
    Y = np.empty(Y_shape, dtype=np.float32)

    # array[internal_cells] == array[ng_prof:ixe0, ng_prof:ixe1, ng_prof:ixe2]
    internal_cells = (slice(ng_prof, ixe[0]),
                      slice(ng_prof, ixe[1]),
                      slice(ng_prof, ixe[2]))

    # internal subdomaon plus just enough ghost cells for smoothing filter and
    # overlapped marching cubes algorithm
    mcubes_stencil = (slice(ng_prof - dx_mcubes - 1, ixe[0] + dx_mcubes + 1),
                      slice(ng_prof - dx_mcubes - 1, ixe[1] + dx_mcubes + 1),
                      slice(ng_prof - dx_mcubes - 1, ixe[2] + dx_mcubes + 1))

    # internal subdomain plus just enough ghost cells for scalar derivative
    deriv_stencil = (slice(ng_prof - ng_akima, ixe[0] + ng_akima),
                     slice(ng_prof - ng_akima, ixe[1] + ng_akima),
                     slice(ng_prof - ng_akima, ixe[2] + ng_akima))

    # default for Read_all is np.float32 on disk, returns np.float32 for work
    U[0][internal_cells] = reader.Read_all(fmt(prefix[0], tstep))
    U[1][internal_cells] = reader.Read_all(fmt(prefix[1], tstep))
    U[2][internal_cells] = reader.Read_all(fmt(prefix[2], tstep))

    rho = reader.Read_all(fmt(prefix[3], tstep))
    T = reader.Read_all(fmt(prefix[4], tstep))
    Y[0] = reader.Read_all(fmt(prefix[5], tstep))
    Y[1] = reader.Read_all(fmt(prefix[6], tstep))
    Y[2] = reader.Read_all(fmt(prefix[7], tstep))
    Y[3] = reader.Read_all(fmt(prefix[8], tstep))
    Y[4] = reader.Read_all(fmt(prefix[9], tstep))
    Y[5] = reader.Read_all(fmt(prefix[10], tstep))
    Y[6] = reader.Read_all(fmt(prefix[11], tstep))
    Y[7] = reader.Read_all(fmt(prefix[12], tstep))

    if comm.rank == 0:
        print('%s: data read from disk' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 2) COMPUTE THERMOCHEMICAL FIELDS
    # ------------------------------------------------------------------
    U[:2] *= 0.01
    rho *= 1000.0
    Y[8] = 1.0 - np.sum(Y[:-1], axis=0)
    P = np.empty_like(T)
    HRR = np.empty_like(T)
    therm = np.empty_like(T)
    hs = np.empty_like(T)

    for k in range(nnx[0]):
        for j in range(nnx[1]):
            for i in range(nnx[2]):
                gas.TDY = T[k, j, i], rho[k, j, i], Y[:, k, j, i]

                cp = gas.cp_mass
                W = gas.mean_molecular_weight
                hwi = gas.partial_molar_enthalpies
                Cdot = gas.net_production_rates

                # pressure
                P[k, j, i] = gas.P

                # heat release rate
                HRR[k, j, i] = - np.sum(Cdot*hwi)

                # thermicity
                therm[k, j, i] = np.sum((W - hwi/(cp*T[k, j, i]))
                                        *Cdot)/rho[k, j, i]

                # standard enthalpy
                gas.TPY = 298.0, ct.one_atm, Y[:, k, j, i]
                hs[k, j, i] = gas.enthalpy_mole

    if comm.rank == 0:
        print('%s: thermochemical fields computed' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 3) EXCHANGE GHOST CELLS (INCLUDING EDGES AND CORNERS)
    # ------------------------------------------------------------------
    U[3][internal_cells] = rho
    U[4][internal_cells] = P
    U[5][internal_cells] = T
    U[6][internal_cells] = Y[iH2]
    U[7][internal_cells] = hs
    U[8][internal_cells] = HRR
    U[9][internal_cells] = therm

    rho = P = T = Y = hs = HRR = therm = None  # free memory

    exchange_ghost_cells(comm, U[:10], nnx, ng_prof)

    if comm.rank == 0:
        print('%s: ghost cells exchanged' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 4) COMPUTE GRADIENT FIELDS AND EXCHANGE GHOST CELLS
    # ------------------------------------------------------------------
    # Dilatation
    div_u_stencil = (slice(0, 3), ) + deriv_stencil  # + concatenates tuples
    U[10][internal_cells] = divergence(U[div_u_stencil], dx)

    # Temperature Gradient
    gradT_cells = (slice(11, 14), ) + internal_cells  # + concatenates tuples
    scalar_gradient(U[5][deriv_stencil], dx, out=U[gradT_cells])

    exchange_ghost_cells(comm, U[10:], nnx, ng_prof)

    if comm.rank == 0:
        print('%s: gradients computed and exchanged' % timeofday(), flush=True)

    # ------------------------------------------------------------------
    # 5) GENERATE THE SMOOTHED ISOSURFACE
    # ------------------------------------------------------------------
    iso = 6  # index for U array for marching cubes isosurface
    iso_field = tophat_filter_3pt(U[iso][mcubes_stencil])
    iso_min = iso_field.min()
    iso_max = iso_field.max()

    if (iso_min < iso_val) and (iso_val < iso_max):
        # compute isosurface vertices and vertex triplets with marching cubes
        try:

            verts, faces = marching_cubes(iso_field, iso_val,
                                          step_size=dx_mcubes,
                                          allow_degenerate=False)[:2]

            # find the face barycenters, unit normal vectors, and surface areas
            centers, normals, areas = isosurface_face_centers(
                                            verts, faces, dx_mcubes, ixe)
            centers += ng_prof - dx_mcubes
            print('MPI rank %d has %d isosurface faces' %
                  (comm.rank, centers.shape[0]), flush=True)

        except RuntimeError:
            print('!! NOTE: MPI rank %d is missing an isosurface' % comm.rank,
                  flush=True)
            centers = np.empty((0, 3), dtype=np.float32)
            normals = np.empty((0, 3), dtype=np.float32)
            areas = np.empty((0,), dtype=np.float32)

    else:
        print('!! NOTE: MPI rank %d is missing an isosurface' % comm.rank,
              flush=True)
        centers = np.empty((0, 3), dtype=np.float32)
        normals = np.empty((0, 3), dtype=np.float32)
        areas = np.empty((0,), dtype=np.float32)

    nfaces = centers.shape[0]
    total_faces = comm.allreduce(nfaces, op=MPI.SUM)
    total_area = comm.allreduce(areas.sum(), op=MPI.SUM)

    if comm.rank == 0:
        print('BEFORE:')
        print('total number of faces = %d' % total_faces)
        print('total surface area / N^2: %f' % (total_area/N**2))
        print('%s: isosurface generated\n' % timeofday(), flush=True)

    centers = np.ascontiguousarray(centers[::skip])
    normals = np.ascontiguousarray(normals[::skip])
    areas = np.ascontiguousarray(areas[::skip])

    nfaces = centers.shape[0]
    total_faces = comm.allreduce(nfaces, op=MPI.SUM)
    total_area = comm.allreduce(areas.sum(), op=MPI.SUM)

    if comm.rank == 0:
        print('AFTER:')
        print('total number of faces = %d' % total_faces)
        print('total surface area / N^2: %f' % (total_area/N**2), flush=True)
    comm.Barrier()

    # ------------------------------------------------------------------
    # 6) INTERPOLATE PROFILES ALONG THE ISOSURFACE NORMALS
    # ------------------------------------------------------------------
    profiles = interpolate_profiles(U, centers, normals, interp_points)

    print('%s: %d - profiles computed with skip %d'
          % (timeofday(), comm.rank, skip), flush=True)

    save_file = '%s/%s-profiles_%03d.npz' % (odir, prob_id, comm.rank)
    np.savez(save_file, profiles=profiles, centers=centers, normals=normals,
             areas=areas, allow_pickle=False)

    # ------------------------------------------------------------------
    # THE END!!!!!!
    # ------------------------------------------------------------------
    return


###############################################################################
def exchange_ghost_cells(comm, U, nnx, ng):
    """
    Empty Docstring!
    """
    # array[all_cells] == array[:, :, :, ...]
    all_cells = (slice(None, None), )*U.ndim
    ixe = nnx + ng

    # spatial dimension
    for dim in range(3):
        # displacement along spatial dimension
        for disp in [-1, 1]:
            tag = dim*disp

            disp_vec = np.zeros(3)
            disp_vec[dim] = disp
            send_rank = comm.Get_cart_rank(comm.coords + disp_vec)
            recv_rank = comm.Get_cart_rank(comm.coords - disp_vec)

            send_slice = list(all_cells)
            recv_slice = list(all_cells)

            # [ng, 2*ng) <-> [nnx+ng, nnx+2*ng)
            if disp == -1:
                send_slice[dim+1] = slice(ng, 2*ng)
                recv_slice[dim+1] = slice(ixe[dim], None)
            # [0, ng) <-> [nnx, nnx+ng)
            else:
                send_slice[dim+1] = slice(nnx[dim], ixe[dim])
                recv_slice[dim+1] = slice(0, ng)

            send_slice = tuple(send_slice)
            recv_slice = tuple(recv_slice)

            gshape = list(U.shape)
            gshape[dim+1] = ng

            recv_ghost_cells = np.empty(gshape, dtype=U.dtype)
            irecv_request = comm.Irecv(recv_ghost_cells, recv_rank, tag)

            send_ghost_cells = np.ascontiguousarray(U[send_slice])
            comm.Send(send_ghost_cells, send_rank, tag)

            irecv_request.wait()
            U[recv_slice] = recv_ghost_cells

    return


def divergence(var, dx=1, ng=3):
    """
    Calculate and return the divergence of a vector field.
    """
    div = flux_diff(var[0], axis=2, ng=ng)  # note the indexing rules
    div+= flux_diff(var[1], axis=1, ng=ng)  # note the indexing rules
    div+= flux_diff(var[2], axis=0, ng=ng)  # note the indexing rules

    div *= 1.0/dx  # change this before adding to mpiToolkit

    return div


def scalar_gradient(var, dx=1, ng=3, out=None):
    """
    Calculate the gradient of a scalar field
    """
    if out is None:
        shape = [nx - 2*ng for nx in var.shape]
        shape.insert(3, 0)
        out = np.empty(shape, dtype=var.dtype)

    out[0] = flux_diff(var, axis=2)  # note the indexing rules
    out[1] = flux_diff(var, axis=1)  # note the indexing rules
    out[2] = flux_diff(var, axis=0)  # note the indexing rules

    out *= 1.0/dx  # change this before adding to mpiToolkit

    return out


@jit(nopython=True, nogil=True)
def flux_diff(var, x=None, axis=0, ng=3):
    """
    1st order difference of interpolated midpoints along given axis of var.
    """
    assert var.ndim == 3, "ERROR: var.ndim not equal to 3!"
    assert ng > 2, "ERROR: a minimum 3 ghost cells on each side are required!"

    nz, ny, nx = var.shape
    out = np.empty((nz-2*ng, ny-2*ng, nx-2*ng), dtype=var.dtype)

    axis = axis % var.ndim
    axes = ((axis+1) % 3, (axis+2) % 3, axis)

    varT = var.transpose(axes)   # new view into the inputs
    outT = out.transpose(axes)   # new view into the outputs

    nx0, nx1, nx2 = varT.shape

    if x is None:
        x = np.arange(nx2).astype(var.dtype)

    # midpoints surrounding internal data points
    xh = 0.5*(x[ng-1:-ng] + x[ng:-ng+1])

    # since axis 2 of varT may not actually be contiguous, use temp array
    for k in range(ng, nx0-ng):
        for j in range(ng, nx1-ng):
            temp = np.ascontiguousarray(varT[k, j])
            varh = interp1d(x, temp, xh)
            outT[k-ng, j-ng, :] = (varh[1:] - varh[:-1])

    return out


@stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def box_kernel_3pt(a):
    cumulant = 0.0
    for k in range(-1, 2):
        for j in range(-1, 2):
            for i in range(-1, 2):
                cumulant += a[k, j, i]
    return cumulant / 27


@jit(nopython=True, nogil=True)
def tophat_filter_3pt(var):
    result = box_kernel_3pt(var)
    return result[1:-1, 1:-1, 1:-1]


def isosurface_face_centers(verts, faces, ixs, ixe):
    nfaces = faces.shape[0]
    centers = np.empty(faces.shape, dtype=np.float32)
    normals = np.empty(faces.shape, dtype=np.float32)
    mask = [True]*nfaces

    for f in range(nfaces):
        v1, v2, v3 = faces[f]
        centers[f] = (verts[v1] + verts[v2] + verts[v3])/3
        vec12 = verts[v2] - verts[v1]
        vec13 = verts[v3] - verts[v1]

        # vector cross product. np.cross is not implemented in numba
        normals[f, 0] = vec12[1]*vec13[2] - vec12[2]*vec13[1]
        normals[f, 1] = vec12[2]*vec13[0] - vec12[0]*vec13[2]
        normals[f, 2] = vec12[0]*vec13[1] - vec12[1]*vec13[0]

        # ignore faces that aren't in the internal subdomain of this task
        mask[f] = np.all(centers[f] >= ixs) and np.all(centers[f] < ixe)

    centers = np.array(centers[mask])
    normals = np.array(normals[mask])

    # compute areas from normal-vector magnitudes
    areas = 0.5*np.sqrt(np.sum(normals**2, axis=1))

    # normalize face-normal vectors to unit vectors
    normals /= np.sqrt(np.sum(normals**2, axis=1, keepdims=True))

    return centers, normals, areas


@jit(nopython=True, nogil=True, parallel=False, cache=True)
def interpolate_profiles(U, centers, normals, interp_points):
    """
    Empty Docstring!
    """
    nvars = U.shape[0]
    nfaces = centers.shape[0]
    # npoints =  + 1
    # interp_points = np.arange+1).reshape(npoints, 1)
    npoints = interp_points.shape[0]

    profiles = np.empty((nvars, nfaces, npoints), dtype=np.float32)

    for f in range(nfaces):
        center = centers[f]
        xi = center + normals[f].reshape(1, 3)*interp_points
        ixc = xi.astype(np.int32)
        ixs = ixc - 2
        ixe = ixc + 4

        for p in prange(npoints):
            is0, is1, is2 = ixs[p]
            ie0, ie1, ie2 = ixe[p]
            x0 = np.arange(is0, ie0)
            x1 = np.arange(is1, ie1)
            x2 = np.arange(is2, ie2)
            data = U[:, is0:ie0, is1:ie1, is2:ie2]

            for v in range(nvars):
                profiles[v, f, p] = interp3d(x0, x1, x2, data[v], xi[p])[0]

    return profiles


@jit(nopython=True, nogil=True, cache=True)
def interp3d(x0, x1, x2, f, xp):
    """
    Returns scalar function value at a single point in three-dimensional
    space using tri-cubic Akima spline interpolation
    """
    # x0 = np.asarray(x0)
    # x1 = np.asarray(x1)
    # x2 = np.asarray(x2)
    # f = np.asarray(f)
    # xp = np.asarray(xp)
    nk, nj, ni = f.shape

    # assert x0.ndim == x1.ndim == x2.ndim == 1  # , 'x0, x1, x2 must be 1D!'
    # assert xp.size == 3  # , 'xp must have size 3!'

    # assert x0.size == nk
    # assert x1.size == nj
    # assert x2.size == ni
    # assert nk == nj and nj == ni and ni == 6

    tk = np.empty(nk, dtype=f.dtype)
    tj = np.empty(nj, dtype=f.dtype)

    for k in range(nk):
        for j in range(nj):
            tj[j] = interp1d(x2, f[k, j], np.array([xp[2]]))[0]
        tk[k] = interp1d(x1, tj, np.array([xp[1]]))[0]

    fi = interp1d(x0, tk, np.array([xp[0]]))

    return fi


@jit(nopython=True, nogil=True, cache=True)
def interp1d(x, f, xi):
    """
    Returns scalar function values at any number of points in
    one-dimensional space using cubic Akima spline interpolation
    """
    # x = np.asarray(x)
    # f = np.asarray(f)
    # xi = np.array([xi]).ravel()

    # assert x.size >= 6
    # assert np.all(xi >= x[2]) and np.all(xi <= x[-3])

    dx = x[1:] - x[:-1]
    m = (f[1:] - f[:-1]) / dx    # m.size = x.size - 1
    dm = np.abs(m[1:] - m[:-1])  # dm.size = x.size - 2
    a = dm[2:]   # a = |m[3]-m[2]|, |m[4]-m[3]|
    b = dm[:-2]  # b = |m[1]-m[0]|, |m[2]-m[1]|
    ab = a + b

    # t.size = x.size - 4, also scipy uses ab > 1e-9*ab.max() for some reason
    t = np.where(ab > 0.0, (a*m[1:-2] + b*m[2:-1]) / ab,
                 0.5*(m[1:-2] + m[2:-1]))

    # p1.size = p2.size = p3.size = x.size - 5
    p1 = t[:-1]
    p2 = (3.0*m[2:-2] - 2.0*t[:-1] - t[1:]) / dx[2:-2]
    p3 = (t[:-1] + t[1:] - 2.0*m[2:-2]) / dx[2:-2]**2

    idx = np.searchsorted(x, xi, side='right') - 1
    w = xi - x[idx]
    p0 = f[idx]
    fmin = np.fmin(f[idx], f[idx+1])
    fmax = np.fmax(f[idx], f[idx+1])

    idx -= 2
    # polynomial version:
    fi = p0 + p1[idx]*w + p2[idx]*w**2 + p3[idx]*w**3

    # fused multiply-add version:
    # fi = p0 + (p1[idx] + (p2[idx] + p3[idx]*w)*w)*w

    # preserve extrema -- not a part of the original Akima spline!!
    fi = np.fmin(fmax, np.fmax(fi, fmin))

    return fi


###############################################################################
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    isosurface_profile_interpolation()
