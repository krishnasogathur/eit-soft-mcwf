# EIT cooling of axial mode of a trapped ion

Quantum Monte Carlo wavefunction (qMCWF) simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode. Files contain the codes used for the Fully Quantum (FQ) simulation results in the paper. (DOI updated soon)

## Definitions

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
