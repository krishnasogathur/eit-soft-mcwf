# EIT cooling of axial mode of a trapped ion

Quantum Monte Carlo wavefunction (qMCWF) simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode. Files contain the codes used for the Fully Quantum (FQ) simulation results in the paper. (DOI updated soon)

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

Requires `numpy`, `scipy` and `joblib`.
