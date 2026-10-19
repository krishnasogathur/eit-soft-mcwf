# EIT cooling of axial mode of a trapped ion

Quantum Monte Carlo wavefunction (qMCWF) simulation of electromagnetically-induced-transparency
cooling of a single ion on one motional mode. Files contain the codes used for the Fully Quantum (FQ) simulation results in the paper. (DOI updated soon)

## Definitions

### Method (SOFT-MCWF)

$$H_{\text{eff}} = H - \frac{i}{2}\sum_k L_k^\dagger L_k$$

$$\psi(t+\varepsilon) = U_p(\varepsilon/2)\,U_x(\varepsilon)\,U_p(\varepsilon/2)\,\psi(t), \qquad U = e^{-iH_{\text{eff}}\,dt/\hbar}$$

$$\|\psi(t)\|^2 < r \;\;\Rightarrow\;\; \text{jump, draw } k \text{ with prob. } \frac{\|L_k\psi\|^2}{\sum_j\|L_j\psi\|^2}$$

`U_p` is diagonal; `U_x` is block-diagonal over the grid, one `n_levels x
n_levels` block per point, each exponentiated by block eigendecomposition. The
jump time within a step is located by root-finding on $\|\psi(t)\|^2 = r$, not
by checking at the end of the step.

### EIT model

$$\eta = k_z x_0, \qquad x_0 = \sqrt{\hbar/m\omega_z}, \qquad \eta_{\text{ph}} = \eta/\sin\theta$$

$$\langle dn\rangle_{\text{per emission}} = \frac{\eta_{se}^2}{2}, \qquad \eta_{se} = \eta_{\text{ph}}\sqrt{x}$$

$$\langle\cos^2\theta\rangle = \tfrac{1}{3}\cdot\tfrac{2}{5} + \tfrac{2}{3}\cdot\tfrac{1}{5} = \frac{4}{15} \qquad (x = \texttt{--se-cos2})$$

$$N \ge \frac{2}{\pi}\,\text{safety}^2\,(2n_{hi}+1) \qquad \text{grid / turning point}$$

$$N \ge 2\,\text{safety}\sqrt{2n_{hi}+1}\;\frac{\text{kdp\_floor}}{\eta} \qquad \text{grid / } k_{\text{beam}}/dp \text{ floor}$$

$$\langle R\rangle(\bar n) = \sum_k P(k;\bar n)\,R(k), \qquad P(k;\bar n) = \frac{\bar n^k e^{-\bar n}}{k!}$$

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
