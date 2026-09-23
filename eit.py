"""EIT cooling of a single trapped ion: one point of the rate curve.

Initialises the motion in a Fock state (or a thermal mixture sampled per
trajectory), propagates an MCWF ensemble, and fits dnbar/dt.

    python -u eit.py --n 100 --t-us 25 --M 2000
    python -u eit.py --n 100 --t-us 25 --M 2000 --se-recoil --se-cos2 4/15

Spontaneous emission is modelled in 1D as a kick of +-k_se along the trap
axis at equal weight.  Since a jump exp(i q x) shifts p by hbar q exactly, at
all orders in the Lamb-Dicke parameter,

    <dn> per emission = (q x0)^2 / 2 = eta_se^2 / 2

for any state.  That matters here because eta_se sqrt(2n+1) reaches ~23 at
n = 950, far outside the Lamb-Dicke regime.

--se-cos2 sets the axial momentum fraction, eta_se = eta_photon sqrt(x).  It
reinterprets the same two-point form as a second-moment match to a 3D
emission pattern: 1/3 isotropic, 2/5 dipole perpendicular to the axis, 1/5
dipole along it.  The physical value for this level scheme is the
branching-weighted average

    <cos^2 theta> = (1/3)(2/5) + (2/3)(1/5) = 4/15,

one third of the decay going to the perpendicular-dipole channel.  With
--se-recoil omitted, decay is kept but the emission kick is dropped.

The output npz holds per-trajectory n(t), not just the ensemble mean, so the
fit can be redone over sub-windows and the error re-derived without recompute.
"""
import argparse, os, shutil, tempfile, time
from contextlib import redirect_stdout
from fractions import Fraction

import numpy as np
from joblib import Parallel, delayed, dump, load

from config import (Delta_R, Gamma_b, Gamma_g, Gamma_e, Omega_g, Omega_e,
                    US_PER_NAT, dAC, eta, eta_ph, hbar, k_beam, m, n_levels,
                    nu_trap, se_coeffs, size_grid, thermal_cap, x0)
from mcwf import (basis, build_KE_hamiltonian, build_fock_state,
                  build_harmonic_potential,
                  build_multilevel_interaction_hamiltonian,
                  build_multilevel_self_hamiltonian, init_arrays,
                  init_nh_hamiltonians, perform_ifft, prepare_nh_evol_ops,
                  sigmaij, simulate_trajectory_block)


def se_wavenumber(axial_frac):
    """Emission wavenumber for the 1D two-direction kick model."""
    return eta_ph*np.sqrt(axial_frac)/x0


def build_ops(N, pmax, eps, se_recoil, se_cos2=1.0):
    """Propagator, jump operators and observables on an N-point grid."""
    x, p, dx, dp = init_arrays(N, pmax)
    I_space, I_NLS = np.ones(N), np.eye(n_levels)

    Delta = [Delta_R, Delta_R, 0]               # two-photon resonance
    exp_mikx = np.exp(-1j*k_beam*x)             # k_{z,1} = +kz
    exp_pikx = np.exp(+1j*k_beam*x)             # k_{z,2} = -kz, counterpropagating

    H_kin = build_KE_hamiltonian(m, p)
    H_nls_block = (build_multilevel_self_hamiltonian(Delta, n_levels=n_levels)[None, :, :]
                   * I_space[:, None, None])
    H_int = (build_multilevel_interaction_hamiltonian([(Omega_g, 0, 2)], exp_mikx=exp_mikx, n_levels=n_levels)
             + build_multilevel_interaction_hamiltonian([(Omega_e, 1, 2)], exp_mikx=exp_pikx, n_levels=n_levels))
    V = build_harmonic_potential(m, nu_trap, x)

    if se_recoil:
        k_se = se_wavenumber(se_cos2)
        c_ops = [np.exp(1j*k_se*mu*x)[:, None, None]
                 * (np.sqrt(se/2)*sigmaij(j, i, n_levels=n_levels))[None, :, :]
                 for se, i, j in se_coeffs for mu in (-1, 1)]
    else:
        c_ops = [np.sqrt(se)*sigmaij(j, i, n_levels=n_levels)[None, :, :]*np.ones((N, 1, 1))
                 for se, i, j in se_coeffs]

    # the damping term and the jumps are built from the SAME set, so the
    # non-Hermitian evolution and the quantum jumps describe one process
    correction_blocks = sum(L.conj().transpose(0, 2, 1) @ L for L in c_ops)
    H_x_nh, H_p_nh = init_nh_hamiltonians(
        H_nls_block + H_int + V[:, None, None]*I_NLS[None, :, :], H_kin, correction_blocks)
    laser_ops = prepare_nh_evol_ops(H_x_nh, H_p_nh, eps, hbar)

    # index map used downstream: 0=V 1=KE 2=<x> 3=<x^2> 4=<p> 5=<p^2> 6,7,8=populations
    e_ops = [V, H_kin, x, x*x, p, p*p,
             sigmaij(0, 0, n_levels), sigmaij(1, 1, n_levels), sigmaij(2, 2, n_levels)]
    mom_array = [False, True, False, False, True, True, False, False, False]
    return x, p, dx, pmax, laser_ops, c_ops, e_ops, mom_array


def edge_weight(psi_p, x, p, pmax, frac=0.9):
    """Weight in the outer (1 - frac) of each box.

    The FFT box is periodic, so anything reaching an edge wraps rather than
    reflects, and a momentum wrap aliases high p into low p, which reads as
    spurious cooling.  This has to be measured, not assumed.
    """
    dp_ = np.abs(psi_p)**2
    dp_ = dp_.sum(axis=0); dp_ /= dp_.sum()
    psi_x = perform_ifft(psi_p)
    dx_ = np.abs(psi_x)**2
    dx_ = dx_.sum(axis=0); dx_ /= dx_.sum()
    Lh = (x[-1] - x[0])/2
    return float(dx_[np.abs(x) > frac*Lh].sum()), float(dp_[np.abs(p) > frac*pmax].sum())


def run_point(n_target, t_us, M, args):
    """Propagate M trajectories at initial occupation n_target and fit the rate."""
    if args.mode == "thermal":
        n_hi, trunc = thermal_cap(n_target, args.tail_tol)
    else:
        n_hi, trunc = int(round(n_target)), 0.0

    N, pmax = size_grid(n_hi, args.safety, kdp_floor=args.kdp_floor)
    n_steps = int(t_us/US_PER_NAT/args.eps)

    t0 = time.perf_counter()
    x, p, dx, pmax, laser_ops, c_ops, e_ops, mom_array = build_ops(
        N, pmax, args.eps, args.se_recoil, args.se_cos2)
    xi = x/x0
    t_build = time.perf_counter() - t0

    # dump once and memmap in the workers, so the operators are not re-pickled
    # per task; at N = 8192 they are the dominant object
    tmpdir = tempfile.mkdtemp(prefix="eit_ops_")
    f_ops, f_c, f_e = (os.path.join(tmpdir, f) for f in ("ops", "cops", "eops"))
    dump(laser_ops, f_ops); dump(c_ops, f_c); dump(e_ops, f_e)
    del laser_ops, c_ops, e_ops

    s_x = ((x[-1] - x[0])/2)/(np.sqrt(2*n_hi + 1)*x0)
    print("  n=%8.1f n_hi=%7d N=%7d s_x=%.2f k/dp=%.1f trunc=%.0e t=%.1fus M=%d steps=%d build=%.1fs"
          % (n_target, n_hi, N, s_x, k_beam/(p[1] - p[0]), trunc, t_us, M, n_steps, t_build), flush=True)

    def one_traj(i):
        rng = np.random.RandomState(args.seed + i)
        np.random.seed(args.seed + i)     # the jump draws use the global RNG
        if args.mode == "thermal":
            q = n_target/(1.0 + n_target)
            n_fock = int(min(rng.geometric(p=1 - q) - 1, n_hi))
        else:
            n_fock = n_hi

        lops = load(f_ops, mmap_mode="r")
        cops = load(f_c,   mmap_mode="r")
        eops = load(f_e,   mmap_mode="r")
        psi_init, _ = build_fock_state(basis(0, n_levels), n_fock, nu_trap, xi, dx, m, hbar)

        with redirect_stdout(open(os.devnull, "w")):
            psi_fin, (avgs, norms, jumps, _) = simulate_trajectory_block(
                psi_init, args.eps, [75, 0, 0, 0, 0, 0, 0, 0],
                lops, None, None, None, None,
                cops, None, eops, mom_array,
                n_steps, hbar, return_psis=False,
                save_interval=args.save_interval, save_psis_interval=n_steps)

        # <n> from the virial sum, exact for a harmonic trap: (<V> + <KE>)/nu - 1/2
        n_t = (np.real(avgs[0]) + np.real(avgs[1]))/nu_trap - 0.5
        ex, ep = edge_weight(psi_fin, x, p, pmax)
        return (n_t.astype(np.float32), np.real(avgs[2:]).astype(np.float32),
                len(jumps), n_fock, ex, ep)

    t0 = time.perf_counter()
    try:
        out = Parallel(n_jobs=args.jobs, backend="loky", verbose=args.progress)(
            delayed(one_traj)(i) for i in range(M))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    t_run = time.perf_counter() - t0

    n_all = np.array([o[0] for o in out])               # (M, nT)
    obs   = np.array([o[1] for o in out]).mean(axis=0)  # (7, nT)
    njump = np.array([o[2] for o in out])
    n0    = np.array([o[3] for o in out])
    edx   = np.array([o[4] for o in out])
    edp   = np.array([o[5] for o in out])

    nT   = n_all.shape[1]
    t_ax = np.arange(nT)*args.eps*args.save_interval*US_PER_NAT      # us
    fit  = t_ax >= args.skip_us                                      # drop the transient
    if fit.sum() < 4:
        fit = np.ones(nT, bool)

    # The OLS slope is linear in the data, so the mean of the per-trajectory
    # slopes equals the slope of the mean curve, and their spread over the
    # ensemble is an honest standard error on it.
    tc     = t_ax[fit] - t_ax[fit].mean()
    slopes = (n_all[:, fit] - n_all[:, fit].mean(axis=1, keepdims=True)) @ tc/np.sum(tc**2)
    rate     = float(slopes.mean())
    rate_sem = float(slopes.std(ddof=1)/np.sqrt(len(slopes)))

    quad = np.polyfit(t_ax[fit], n_all[:, fit].mean(axis=0), 2)
    curv = float(abs(quad[0]*t_ax[-1]**2)/max(abs(rate*t_ax[-1]), 1e-12))

    warn = ""
    if max(edx.max(), edp.max()) > 1e-8:
        warn = "  ** EDGE x=%.1e p=%.1e -- possible wrap" % (edx.max(), edp.max())
    if trunc > 1e-2:
        warn += "  ** TRUNC %.1e" % trunc
    print("      n(0)=%8.1f  dn/dt=%+.5f +- %.5f quanta/us  (%+.4g /s)  jumps=%.1f  curv=%.2f  run=%.0fs%s"
          % (n_all[:, 0].mean(), rate, rate_sem, rate*1e6, njump.mean(), curv, t_run, warn), flush=True)

    return dict(
        n_target=n_target, mode=args.mode, n_hi=n_hi, trunc_weight=trunc,
        N=N, pmax=pmax, eps=args.eps, t_us=t_us, M=M, safety=args.safety,
        se_recoil=args.se_recoil, se_cos2=args.se_cos2,
        eta_se=(eta_ph*np.sqrt(args.se_cos2) if args.se_recoil else 0.0),
        skip_us=args.skip_us,
        t_axis=t_ax, n_all=n_all, n0=n0, njump=njump, edge_x=edx, edge_p=edp,
        obs_t=obs, obs_names=np.array(["x", "x2", "p", "p2", "pop0", "pop1", "pop2"]),
        slopes=slopes, rate=rate, rate_sem=rate_sem,
        rate_per_s=rate*1e6, rate_per_s_sem=rate_sem*1e6, curv_frac=curv,
        eta=eta, nu_trap=nu_trap, Gamma_b=Gamma_b, dAC=dAC,
        t_build=t_build, t_run=t_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=float, required=True,
                    help="initial motional occupation")
    ap.add_argument("--t-us", type=float, required=True,
                    help="evolution time in microseconds")
    ap.add_argument("--M", type=int, required=True,
                    help="number of trajectories")
    ap.add_argument("--mode", choices=["fock", "thermal"], default="fock",
                    help="definite n, or a thermal mixture of mean n sampled per trajectory")
    ap.add_argument("--eps", type=float, default=0.1,
                    help="propagation step in Gamma_1^-1")
    ap.add_argument("--skip-us", type=float, default=2.0,
                    help="startup transient dropped before fitting")
    ap.add_argument("--save-interval", type=int, default=50,
                    help="steps between stored observables")
    ap.add_argument("--safety", type=float, default=None,
                    help="grid margin on the turning point (default: config.SAFETY)")
    ap.add_argument("--kdp-floor", type=float, default=None,
                    help="minimum k_beam/dp; 0 disables (default: config.KDP_FLOOR)")
    ap.add_argument("--tail-tol", type=float, default=1e-3,
                    help="thermal mode: Fock mass allowed beyond the cutoff")
    ap.add_argument("--se-recoil", action="store_true",
                    help="restore the +-k emission kick (default: decay without recoil)")
    ap.add_argument("--se-cos2", type=str, default="1.0",
                    help="axial momentum fraction; accepts a fraction, e.g. 4/15")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--progress", type=int, default=0,
                    help="joblib verbosity; >0 reports trajectories as they finish")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default=None,
                    help="output npz (default: eit_n<n>.npz)")
    args = ap.parse_args()

    import config
    args.se_cos2 = float(Fraction(args.se_cos2))
    if args.safety is None:
        args.safety = config.SAFETY
    if args.kdp_floor is None:
        args.kdp_floor = config.KDP_FLOOR

    print("=== EIT cooling rate ===")
    print("  eta = %.6f   nu_trap = %.4f   Gamma_b = %.5f" % (eta, nu_trap, Gamma_b))
    print("  AC Stark shift = %.4f vs nu_trap = %.4f (ratio %.4f)" % (dAC, nu_trap, dAC/nu_trap))
    print("  mode=%s  SE recoil=%s" % (args.mode, "ON" if args.se_recoil else "OFF"))
    if args.se_recoil:
        eta_se = eta_ph*np.sqrt(args.se_cos2)
        print("  SE: 1D +-k kick, axial_frac=%.4f  eta_se=%.4f (= %.2f x eta_laser)  dn/emission=%.5f"
              % (args.se_cos2, eta_se, eta_se/eta, eta_se**2/2))
    print(flush=True)

    res = run_point(args.n, args.t_us, args.M, args)
    out = args.out or "eit_n%.1f.npz" % args.n
    np.savez_compressed(out, **res)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
