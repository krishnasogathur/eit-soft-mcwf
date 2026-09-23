# EIT cooling of a trapped ion

Monte Carlo wavefunction simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode, giving the cooling rate
`dn/dt` as a function of occupation `n`.

The motion is represented on a momentum grid rather than in a truncated Fock
basis, so nothing assumes the Lamb-Dicke limit and occupations of many hundreds
of quanta are reachable.

## Files

| file | |
|---|---|
| `config.py` | physical parameters, natural units, and grid sizing |
| `mcwf.py` | split-operator MCWF propagation of the multilevel atom |
| `eit.py` | the EIT model and one point of the rate curve |
| `binning.py` | Fock-basis rates recombined into a Poissonian mean |

## Use

```
python -u eit.py --n 100 --t-us 25 --M 2000
python -u eit.py --n 100 --t-us 25 --M 2000 --se-recoil --se-cos2 4/15
```

One point per invocation; each writes an `npz` holding per-trajectory `n(t)`,
jump counts and grid-edge weights alongside the fitted rate, so the fit can be
redone over sub-windows without recomputing.

Requires `numpy`, `scipy` and `joblib`.
