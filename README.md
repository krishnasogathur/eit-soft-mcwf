# EIT cooling of axial mode of a trapped ion

Quantum Monte Carlo wavefunction (qMCWF) simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode. Files contain the codes used for the Fully Quantum (FQ) simulation results in the paper. (DOI updated soon)

## Definitions

### Method (SOFT-MCWF)

    H_eff = H - (i/2) sum_k L_k^dag L_k

    psi(t+eps) = U_p(eps/2) U_x(eps) U_p(eps/2) psi(t),    U = exp(-i H_eff dt/hbar)

    ||psi(t)||^2 < r  ->  jump, draw k with prob ||L_k psi||^2 / sum_j ||L_j psi||^2

U_p is diagonal; U_x is the dense internal-level block, exponentiated by
eigendecomposition. The jump time within a step is located by root-finding on
||psi(t)||^2 = r, not by checking at the end of the step.

### EIT model

    eta    = k_z x0,          x0 = sqrt(hbar/m omega_z)
    eta_ph = eta/sin(theta)

    <dn> per emission = eta_se^2 / 2,      eta_se = eta_ph sqrt(x)

    <cos^2 theta> = (1/3)(2/5) + (2/3)(1/5) = 4/15     (x = --se-cos2)

    N >= (2/pi) safety^2 (2 n_hi + 1)                  grid / turning point
    N >= 2 safety sqrt(2 n_hi + 1) kdp_floor / eta      grid / k_beam-dp floor

    <R>(nbar) = sum_k P(k; nbar) R(k),     P(k; nbar) = nbar^k e^-nbar / k!

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
