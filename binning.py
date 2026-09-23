"""Fock-basis R(n) -> Poissonian <R>(nbar) by binning.

    from binning import recombine
    nbar, R_poisson = recombine(n, R_fock)

    python binning.py rates.txt        # two columns, n and R

The simulation is run at definite motional occupation, but a coherent state of
mean nbar has Poissonian occupation, so the observable rate is

    <R>(nbar) = sum_k P(k; nbar) R(k),    P(k; nbar) = nbar^k e^-nbar / k!

R is known only at the scattered n where the simulation was run.  Each integer
k is therefore assigned to the nearest measured point and the probability
inside each cell is summed,

    w_j = sum over k in cell j of P(k; nbar),     <R>(nbar) = sum_j w_j R_j

so that no value of R is ever interpolated or otherwise synthesised.  Values of
nbar with more than `tol` of the Poisson mass outside the measured range are
dropped rather than extrapolated.
"""
import numpy as np
from scipy.stats import poisson


def poisson_kernel(nbar, wide=8.0):
    """Integers within +-wide sigma of nbar, and the normalised pmf there."""
    lo = max(int(nbar - wide*np.sqrt(nbar) - wide), 0)
    k = np.arange(lo, int(nbar + wide*np.sqrt(nbar) + wide) + 1)
    p = poisson.pmf(k, nbar)
    return k, p/p.sum()


def weights(n, nbar):
    """Weight on each measured point, and the leaked Poisson mass."""
    k, p = poisson_kernel(nbar)
    leak = p[(k < n[0]) | (k > n[-1])].sum()
    j = np.abs(np.clip(k, n[0], n[-1])[:, None] - n[None, :]).argmin(1)
    w = np.bincount(j, p, minlength=len(n))
    return w/w.sum(), leak


def recombine(n, R, nbar=None, tol=1e-3):
    """(nbar, R).  `nbar` defaults to the measured n."""
    n = np.asarray(n, float); R = np.asarray(R, float)
    o = np.argsort(n); n, R = n[o], R[o]
    nbar = n if nbar is None else np.atleast_1d(np.asarray(nbar, float))
    keep, V = [], []
    for b in nbar:
        w, leak = weights(n, b)
        if leak > tol:
            continue
        keep.append(b); V.append(w @ R)
    return np.array(keep), np.array(V)


if __name__ == "__main__":
    import sys
    n, R = np.loadtxt(sys.argv[1], unpack=True)
    nbar, Rp = recombine(n, R)
    for b, v in zip(nbar, Rp):
        print("%10.3f %14.6e" % (b, v))
