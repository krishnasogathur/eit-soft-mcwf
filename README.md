# EIT cooling of axial mode of a trapped ion

Quantum Monte Carlo wavefunction (qMCWF) simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode. Files contain the codes used for the Fully Quantum (FQ) simulation results in the paper. (DOI updated soon)

## Definitions

**Lamb-Dicke parameters.** `eta = k_z x0` for the laser wavenumber projected onto
the trap axis, and `eta_ph = k_photon x0` for the full photon wavenumber
(`eta_ph = eta/sin(theta)`), with `x0 = sqrt(hbar/m omega_z)`.

**Emission recoil.** A spontaneous-emission jump is modelled in 1D as a kick of
`+-k_se` along the trap axis, equal weight either way. Since the jump operator
`exp(i q x)` shifts momentum by `hbar q` exactly, at all orders in the
Lamb-Dicke parameter,

    <dn> per emission = (q x0)^2 / 2 = eta_se^2 / 2

for any motional state - this holds even where `eta_se sqrt(2n+1)` is far
outside the Lamb-Dicke regime (~23 at n = 950).

`eta_se = eta_ph sqrt(x)`, where `x` is the axial momentum fraction
(`--se-cos2`). `x = 1` is the full 1D kick model; other values reinterpret the
same two-point form as a second-moment match to a 3D dipole pattern,
`x = <cos^2 theta>`: 1/3 isotropic, 2/5 perpendicular dipole, 1/5 axial dipole.
The physical value for this level scheme is the branching-weighted average

    <cos^2 theta> = (1/3)(2/5) + (2/3)(1/5) = 4/15.

**Grid sizing.** The momentum grid is sized to contain the classical turning
point `sqrt(2 n_hi + 1)` in both `x` and `p` with margin `safety`, and
separately floored so the laser wavenumber resolution `k_beam/dp` stays above
`kdp_floor` (independent of the recoil flag - the lasers transfer `+-k` per
photon regardless of whether emission carries a kick).

**Poisson recombination.** The simulation runs at definite occupation `n`, but
a phase-averaged coherent (PAC) state of mean `nbar` has Poissonian occupation,
so the observable rate is `<R>(nbar) = sum_k P(k; nbar) R(k)`. `binning.py`
assigns each integer `k` to the nearest measured `n_j` and sums the probability
inside each cell, so no value of `R` is ever interpolated.

## Files

| file | |
|---|---|
| `config.py` | physical parameters, natural units, and grid sizing |
| `mcwf.py` | split-operator MCWF propagation of the multilevel atom |
| `eit.py` | the EIT model and one point of the rate curve |
| `binning.py` | Fock-basis rates recombined into a Poissonian mean |
| `plots.ipynb` | the figures, built up curve by curve, including the binning |
| `data/` | the rate curves the notebook plots, as `.npy` |

## Use

```
python -u eit.py --n 100 --t-us 25 --M 2000
python -u eit.py --n 100 --t-us 25 --M 2000 --se-recoil --se-cos2 4/15
```

The notebook reads `data/` and needs nothing else run first.

Requires `numpy`, `scipy` and `joblib`, and `matplotlib` for the notebook.
