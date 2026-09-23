"""Split-operator MCWF propagation of a multilevel atom on a momentum grid.

The state is (n_levels, N) complex, stored in momentum space.  One step of
length `eps` is the Strang splitting

    exp(-i H_p eps/2) exp(-i H_x eps) exp(-i H_p eps/2)

where H_x carries the trap potential, the internal Hamiltonian and the laser
coupling as a dense (N, n_levels, n_levels) block, and H_p is diagonal.  The
internal block is exponentiated by eigendecomposition at every grid point
rather than by a series or a general matrix exponential: a detuning never
commutes with its own coupling, so the internal block must not be split.

Dissipation follows the standard wavefunction Monte Carlo: propagation under
the non-Hermitian H - (i/2) sum c^dag c until the norm drops below a uniform
random r, the crossing time located by Brent root-finding inside the step, a
jump applied by drawing among the c_ops with weight ||c psi||^2, and the
remainder of the step completed with a fresh r.
"""
import numpy as np
from scipy.optimize import brentq


def init_arrays(N, pmax): #no numpy here
    p = np.linspace(-pmax,pmax,N)
    # p = np.arange(-pmax, pmax, dp)
    x = np.fft.fftfreq(N, d=p[1]-p[0]) * 2 * np.pi 
    x = np.fft.fftshift(x) 
    dx = (x[1]-x[0])

    # p = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi) 

    dp = p[1]-p[0]

    return x, p, dx, dp


def basis(idx, n_levels):
    return np.eye(n_levels)[:, idx]


def sigmaij(idx,jdx, n_levels):
    return np.outer(basis(idx, n_levels) , basis(jdx, n_levels))


def perform_fft(psi_x_in, batched=False):
    """Transform from position to momentum basis.
    
    Input shape:
        - if batched: (n_levels x N, ...) where ... can be any shape
        - else:       (n_levels x N,)
    """

    N = psi_x_in.shape[1]

    if batched:
       
        # Apply FFT along the pos axis (axis=1)
        psi_p_out = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(psi_x_in, axes=1), axis=1),
            axes=1
        ) / np.sqrt(N)
        
        # Reshape back to original shape (n_levels x N, ...)
        return psi_p_out
    
    else:
        psi_p_out = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(psi_x_in, axes=1), axis=1),
            axes=1
        ) / np.sqrt(N)
        return psi_p_out


def perform_ifft(psi_p_in, batched=False):
    """Transform from momentum to position basis (with correct shifting).
    Input shape:
        - if batched: (n_levels x N, T)
        - else:       (n_levels x N,)
    """
    N = psi_p_in.shape[1]

    if batched:
        psi_x_out = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(psi_p_in, axes=1), axis=1),
            axes=1
        ) * np.sqrt(N)
        # return psi_x_out.reshape(2 * N, -1)
        return psi_x_out
    else:
        psi_x_out = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(psi_p_in, axes=1), axis=1),
            axes=1
        ) * np.sqrt(N)
        return psi_x_out


def compute_expectations(psi, ops, mom_space):
    psi = psi / np.linalg.norm(psi)
    n_lev = psi.shape[0]                     # ← add

    if not np.all(mom_space):
        psi_pos = perform_ifft(psi)

    vals = np.zeros(len(ops))

    for i, op in enumerate(ops):
        if op.ndim == 1:
            wf = psi if mom_space[i] else psi_pos
            op_psi = wf * op
            vals[i] = np.vdot(wf, op_psi).real

        elif op.ndim == 2 and op.shape == (n_lev, n_lev):   # ← was (2,2)
            op_psi = op @ psi
            vals[i] = np.vdot(psi, op_psi).real

        else:
            raise ValueError(f"Unsupported op shape {op.shape}")

    return vals


def evol_ops(t, H_x_nh_eigs_block, H_p_nh_eigs, hbar=1.0):
    """
    Build evolution operators from precomputed eigenvalues/eigenvectors.

    H_x_nh_block : (N,2) array of eigenvalues of each 2x2 block of Non-hermitian hamilotnian
    """
    
    U_x_nh_eigs_block = np.exp(-1j * H_x_nh_eigs_block * t / hbar)   # shape (2,N)
    U_p_nh_eigs = np.exp(-1j * H_p_nh_eigs * t / (2*hbar))


    return U_x_nh_eigs_block, U_p_nh_eigs


def block_eigendecomp(H_blocks):
    """
    Perform eigendecomposition of a block-diagonal Hamiltonian.

    Parameters
    ----------
    H_blocks : ndarray, shape (N,2,2)
        Block-diagonal Hamiltonian (2x2 per block)

    Returns
    -------
    eigvals : ndarray, shape (N,2)
        Eigenvalues of each 2x2 block
    V : ndarray, shape (N,2,2)
        Eigenvectors of each 2x2 block
    V_inv : ndarray, shape (N,2,2)
        Inverse of eigenvectors for each block
    """
    N = H_blocks.shape[0]
    n_levels = H_blocks.shape[1]
    eigvals = np.empty((n_levels,N), dtype=np.complex128)
    V = np.empty((N,n_levels,n_levels), dtype=np.complex128)
    V_inv = np.empty((N,n_levels,n_levels), dtype=np.complex128)

    for n in range(N):
        vals, vecs = np.linalg.eig(H_blocks[n])
        eigvals[:, n] = vals
        V[n] = vecs
        V_inv[n] = np.linalg.inv(vecs)

    # thought about the order of eigenvals for a while; it makes most sense to store it as 2,N (as is our initial state psi)

    return eigvals, V, V_inv 


def prepare_nh_evol_ops(H_x_nh_block, H_p_nh_diag, eps, hbar):
    """
    Given non-Hermitian position and momentum Hamiltonians, compute the
    evolution operators and return relevant decompositions.

    Returns the list:
            [U_x_nh_eps, U_p_nh_eps,
             H_x_nh_eigs, V_x_nh, V_inv_x_nh,
             H_p_nh_diag]
    """
    # Eigendecomposition
    H_x_nh_eigs_block, V_x_nh, V_inv_x_nh = block_eigendecomp(H_x_nh_block)

    # H_p_nh_eigs: assumed to be 1D array (diag of the matrix)

    # Time evolution operators using the user-defined evol_ops function
    U_x_nh_eps_eigs_block, U_p_nh_eps_eigs = evol_ops(eps, H_x_nh_eigs_block, H_p_nh_diag, hbar) 

    # Reconstruct full U_x_nh_eps from eigendecomposition
    U_x_nh_eps_block = np.einsum('nij,jn,njk->nik', V_x_nh, U_x_nh_eps_eigs_block, V_inv_x_nh) 
    # above einsum is non-trivial actually; lemme bc I want to maintain eigs as 2,N for easy element-wise product

    '''  
    U_x_nh_eps (Nx2x2), U_p_nh_eps_eigs (Nx1): used in the general SOFT step (perform_soft_trajectories)
    H_x_nh_eigs, V_x_nh, V_inv_x_nh, H_p_nh_diag: used in sub_step_evol for brentq input
    '''
    return [U_x_nh_eps_block, U_p_nh_eps_eigs,
            H_x_nh_eigs_block, V_x_nh, V_inv_x_nh,
            H_p_nh_diag]


def act_block_on_state(U_blocks, psi, large=True):
    """
    Apply a block-diagonal operator to a state vector or batched state vectors.

    Parameters
    ----------
    U_blocks : ndarray, shape (N, n, n)
        Block-diagonal operator (n x n per block)
    psi : ndarray, shape (n, N)
        State vector
    large : bool
        If True, use reshaped matmul (faster for large N)

    Returns
    -------
    psi_out : ndarray, shape (n, N)
        Resulting state vector after applying U_blocks
    """
    if large:
        # reshape psi for batched matmul: (N, n, 1)
        psi_exp = psi.T[:, :, None]          # (N, n, 1)
        tmp = np.matmul(U_blocks, psi_exp)[:, :, 0].T
        psi_out = tmp
    else:
        # einsum version for small sizes
        psi_out = np.einsum('nij,jn->in', U_blocks, psi)

    return psi_out


def act_block_on_state_optim(U_blocks, psi, large=True):
    """
    Apply a block-diagonal operator to a state vector or batched state vectors.

    Parameters
    ----------
    U_blocks : ndarray, shape (N, n, n)
        Block-diagonal operator (n x n per block)
    psi : ndarray, shape (n, N)
        State vector
    large : bool
        If True, use reshaped matmul (faster for large N)

    Returns
    -------
    psi_out : ndarray, shape (n, N)
        Resulting state vector after applying U_blocks
    """
    if large:
        # reshape psi for batched matmul: (N, n, 1)
        psi_exp = psi.T[:, :, None]          # (N, n, 1)
        tmp = np.matmul(U_blocks, psi_exp)[:, :, 0].T
        psi_out = tmp
    else:
        # einsum version for small sizes
        psi_out = np.einsum('nij,jn->in', U_blocks, psi)

    return psi_out


def perform_soft_trajectories(psi_in, U_x_nh_eps, U_x_nh_eigs, V_x, V_inv_x, U_p_diag, optim=True):
   
    psiout = psi_in
    # Half potential:
    psiout = U_p_diag[None, :] * psiout # already matches along the motional dimension

    psi_ifft = perform_ifft(psiout)

    if optim==False:
        psi_ifft = act_block_on_state(U_x_nh_eps, psi_ifft)  # if not optimizing, I simply act the pre-constructed U_x_nh_eps
    else:
        psi_ifft = act_block_on_state(V_inv_x, psi_ifft)       # Transform to eigenbasis
        psi_ifft = U_x_nh_eigs * psi_ifft        # Both U_x_nh_eigs and psi_ifft are 2xN so elementwise multiplication works
        psi_ifft = act_block_on_state(V_x, psi_ifft) 

    psiout = perform_fft(psi_ifft)

    # Half potential the second time
    psiout = U_p_diag[None, :] * psiout        

    return psiout


def perform_soft_trajectories_optim(psi_in, U_x_nh_eps, U_x_nh_eigs, V_x, V_inv_x, U_p_diag, optim=True):
   
    psiout = psi_in
    # Half potential:
    psiout = U_p_diag[None, :] * psiout # already matches along the motional dimension

    psi_ifft = perform_ifft(psiout)

    if optim==False:
        psi_ifft = act_block_on_state_optim(U_x_nh_eps, psi_ifft)  # if not optimizing, I simply act the pre-constructed U_x_nh_eps
    else:
        psi_ifft = act_block_on_state_optim(V_inv_x, psi_ifft)       # Transform to eigenbasis
        psi_ifft = U_x_nh_eigs * psi_ifft        # Both U_x_nh_eigs and psi_ifft are 2xN so elementwise multiplication works
        psi_ifft = act_block_on_state_optim(V_x, psi_ifft) 

    psiout = perform_fft(psi_ifft)

    # Half potential the second time
    psiout = U_p_diag[None, :] * psiout        

    return psiout


def sub_step_evol(t, psi, r, U_x_nh_eps, H_x_nh_eigs, V_x_nh, V_inv_x_nh, H_p_nh_eigs, hbar=1, return_norm=False): 
    U_x_nh_eigs, U_p_nh_eigs = evol_ops(t, H_x_nh_eigs, H_p_nh_eigs, hbar ) # if necessary, switch to linear later 
    psi_out = perform_soft_trajectories_optim(psi, U_x_nh_eps, U_x_nh_eigs, V_x_nh, V_inv_x_nh, U_p_nh_eigs, optim=True) # all evol ops are diagonal, so multiplication is easie

    if return_norm==True:
        return np.linalg.norm(psi_out)**2-r
    return psi_out


def find_jump_time(psi_prev, eps, r,  U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar=1):
    """
    Finds the sub-time t* ∈ [0, eps] where the evolved psi_prev's norm matches r.
    """
    # print("prev norm inside find jump time func:", np.linalg.norm(psi_prev) **2)

    val0 = sub_step_evol(0, psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True ) 
    val1 = sub_step_evol(eps, psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True )

    if val0 * val1 > 0:
        raise ValueError("Jump not bracketed in [0, eps] — this shouldn't happen if norm detection was correct.")

    t_star = brentq(sub_step_evol, 0, eps, args=(psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True ), xtol=1e-10)
    return t_star


def perform_jump(psi, c_ops):

   
    # (since jump operator is in position space we first convert)
    psiout = perform_ifft(psi)

    # Instead of computing matmul twice, I will do it once and just store the psis. seems much faster this way
    # I'll just candidates + their squared norms in one pass
    psi_candidates = []
    norms = []

    for op in c_ops:
        candidate_state = act_block_on_state(op, psiout)
        norm_candidate = np.linalg.norm(candidate_state)
        psi_candidates.append(candidate_state)
        norms.append(norm_candidate)
    
    norms = np.array(norms)
    probs = norms**2
    probs /= probs.sum()

  
    # Choose jump according to probs
    chosen_index = np.random.choice(len(c_ops), p=probs)

    # Normalize chosen candidate with precomputed norm
    psi = psi_candidates[chosen_index] / norms[chosen_index]
    
    # Transform back to momentum space 
    psiout = perform_fft(psi)
    return psiout


def detect_jump_and_update(psi_curr, eps, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, c_ops, hbar=1):
    t_star = find_jump_time(psi_curr, eps, r,  U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar)
    # print(f"Jump at time step {i}, t* = {t_star:.4e}")
    psi_updated = sub_step_evol(t_star, psi_curr ,r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar) #this is the time at which it gets projected, hence we evolve until norm reaches that point
    psi_after_jump = perform_jump(psi_updated, c_ops)

    return psi_after_jump, t_star


def finish_eps_evol(del_t, psi_after_jump, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar=1):
    psi_eps = sub_step_evol(del_t, psi_after_jump, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar) # evolving the updated state for rest of the time, given by eps-t_curr
    return psi_eps


def build_fock_state(psi_nls, n_fock, nu_trap, xi, dx, m=1, hbar=1): # by default init in momentum space
    n_psi = psi_nls.shape[0]

    # def fock_n(n, x, nu_trap, m=1, hbar=1):
        
    #     """Returns the n-th Fock state position space wavefunction for QHO."""
    #     # coeff = ((m * nu_trap / (np.pi * hbar)) ** (1 / 4)) / np.sqrt(factorial(n)) / (2 ** (n / 2))

    #     log_coeff = (
    #         0.25 * np.log(m * nu_trap / (np.pi * hbar))
    #         - 0.5 * (n * np.log(2) + gammaln(n + 1))
    #     )

    #     coeff = np.exp(log_coeff)

    #     # poly = hermite(n)
    #     # gaussian = np.exp(-x ** 2 / 2)
    #     # return coeff * poly(x) * gaussian
    

    #     # poly = hermite(n)
    #     gaussian = np.exp(-x ** 2 / 2)

    #     gaussian = np.exp(-x**2/2)
    #     H = eval_hermite(n, x)
    #     psi = coeff * H * gaussian

    #     print("coeff =", coeff)
    #     print("H finite:", np.isfinite(H).all())
    #     print("psi finite:", np.isfinite(psi).all())
    #     print("max |H| =", np.nanmax(np.abs(H)))
    #     print("max |psi| =", np.nanmax(np.abs(psi)))
    #     print("norm =", np.linalg.norm(psi))

    #     return psi

    #     return coeff * eval_hermite(n, x) * gaussian
    # def fock_n(n, xi, nu_trap=1.0, m=1, hbar=1):
    #     """n-th normalised HO wavefunction. xi = x/x0. Stable to n ~ 2000+."""
    #     coeff = (m * nu_trap / (np.pi * hbar))**0.25
    #     if n == 0:
    #         return coeff * np.exp(-xi**2 / 2)
    #     if n == 1:
    #         return coeff * np.sqrt(2.0) * xi * np.exp(-xi**2 / 2)

    #     logoff = np.zeros_like(xi)      # per-point log offset
    #     psi0 = np.ones_like(xi)
    #     psi1 = np.sqrt(2.0) * xi

    #     for k in range(1, n):
    #         psi2 = np.sqrt(2.0/(k+1)) * xi * psi1 - np.sqrt(k/(k+1)) * psi0
    #         psi0, psi1 = psi1, psi2
    #         a = np.abs(psi1)
    #         big = a > 1e100
    #         if np.any(big):
    #             f = np.where(big, a, 1.0)
    #             psi0 = psi0 / f
    #             psi1 = psi1 / f
    #             logoff = logoff + np.where(big, np.log(f), 0.0)

    #     lg = np.log(np.abs(psi1) + 1e-300) + logoff - xi**2/2 + np.log(coeff)
    #     return np.sign(psi1) * np.exp(lg)

    def fock_n(n, x, nu_trap, m=1, hbar=1):
            coeff = (m * nu_trap / (np.pi * hbar))**0.25
            if n == 0: return coeff * np.exp(-x**2 / 2)
            if n == 1: return coeff * np.sqrt(2.0) * x * np.exp(-x**2 / 2)
            logoff = np.zeros_like(x); psi0 = np.ones_like(x); psi1 = np.sqrt(2.0) * x
            for k in range(1, n):
                psi2 = np.sqrt(2.0/(k+1)) * x * psi1 - np.sqrt(k/(k+1)) * psi0
                psi0, psi1 = psi1, psi2
                a = np.abs(psi1); big = a > 1e100
                if np.any(big):
                    f = np.where(big, a, 1.0)
                    psi0 = psi0 / f; psi1 = psi1 / f
                    logoff = logoff + np.where(big, np.log(f), 0.0)
            lg = np.log(np.abs(psi1) + 1e-300) + logoff - x**2/2 + np.log(coeff)
            return np.sign(psi1) * np.exp(lg)

    psi_x_init = fock_n(n_fock, xi, nu_trap, m , hbar)  # position space wavefunction in the n-th Fock state
    psi_x_init = psi_x_init * np.sqrt(dx) #this is the psi which satisfies sum(abs(psi)^2) = 1, and useful to preserve normalization in our wavefunction

    psi_nls = psi_nls.ravel()
    # print(np.linalg.norm(psi_nls)  , "norm of psi_nls before normalization")
    # exit(0)
    psi_nls = psi_nls / np.linalg.norm(psi_nls)  # normalized; so that sum(psi(x)^2 dx) = 1 (integral as a discrete sum)
    psi_init_x = np.kron(psi_x_init, psi_nls)  # |g⟩ ⊗ |ψ(x)⟩ #positional state tensored with atomic state for complete state description 


    psi_init_x = psi_init_x.reshape(n_psi,-1, order='F') # need a 2,N array whose 0 corresponds to e and 1 corresponds to g; np.kron() in opp dir messes this up
    psi_init_x /= np.linalg.norm(psi_init_x)  # Normalizing the combined state, so that sum(abs(psi)^2) = 1 (integral as a discrete sum)

    psi_init = perform_fft(psi_init_x)

    return psi_init, psi_init_x


def build_KE_hamiltonian(mass, p):
    KE = (p**2 / (2 * mass))
    return KE


def build_harmonic_potential(mass, nu_trap, x):
    """
    Constructs a harmonic potential Hamiltonian:
        V(x) = (1/2) * m * ω² * x²
    """
    V = 0.5 * mass * ((nu_trap)**2) * (x**2)
    return V


def build_multilevel_self_hamiltonian(Delta_list, n_levels):
    """
    Delta_list: list of tuples (Delta, i)
    """
    H = 0

    for i, Deltai in enumerate(Delta_list):
        H += Deltai * sigmaij(i, i, n_levels)   # diagonal detuning term

    return H


def build_multilevel_interaction_hamiltonian(Omega_list, exp_mikx, n_levels):
    """
    Omega_list: list of tuples (Omega, i, j)
    """
    H = 0

    for Omegaij, i, j in Omega_list:
        H += -(Omegaij/2) * exp_mikx[:, None, None] * (sigmaij(i, j, n_levels) [None, :, :] )  # off-diagonal coupling term

    H = H + np.conjugate(np.transpose(H, (0, 2, 1)))

    return H


def init_nh_hamiltonians(H_x, H_p, correction_blocks):
    """
    Construct non-Hermitian Hamiltonians by adding -i/2 * L†L terms to H_x/.
    """
    # Build non-Hermitian contribution from collapse ops
    # correction = sum(L.conj().T @ L for L in c_ops)
    H_x_nh = H_x - (1j / 2) * correction_blocks
    
    return H_x_nh, H_p


def simulate_trajectory_block(
    psi_input, eps, pulse_timings,
    laser_ops, counter_ops,
    cooling_ops, cooling_ops_counter,
    noint_ops,
    c_ops_lasers, c_ops_cooling,
    e_ops, mom_array,
    n_steps,
    hbar=1,
    return_psis=False,
    save_interval=1,
    save_psis_interval=100   
):
    psip = psi_input
    # Downsampling factor for psip_store 
    downsample_factor = save_interval
    n_store_steps = (n_steps + downsample_factor - 1) // downsample_factor

    downsample_factor_psi = save_psis_interval
    n_store_steps_psi = (n_steps + downsample_factor_psi - 1) // downsample_factor_psi

    avg_vals = np.zeros((len(e_ops) , n_store_steps))
    
    norms_arr = np.zeros(n_store_steps)

    jump_times = []


    # es_projs = np.zeros((N, n_store_steps))

    if return_psis:
        psi_return = np.zeros(psip.shape + (n_store_steps_psi,), dtype=complex)
    else:
        psi_return = None

    store_idx = 0
    store_idx_psi = 0

    if len(pulse_timings) != 8:
        raise ValueError(
            "pulse_timings must be a tuple of 8 values: "
            "(pulse_duration_laser, wait_time_1, pulse_duration_cooling_1, wait_time_2, "
            " pulse_duration_counter, wait_time_3, pulse_duration_cooling_2, wait_time_4)"
        )

    (laser_pulse, wait_time_1, cooling_pulse_1, wait_time_2,
     counter_pulse, wait_time_3, cooling_pulse_2, wait_time_4) = pulse_timings

    # Cumulative segment edges
    t1 = laser_pulse
    t2 = t1 + wait_time_1
    t3 = t2 + cooling_pulse_1
    t4 = t3 + wait_time_2
    t5 = t4 + counter_pulse
    t6 = t5 + wait_time_3
    t7 = t6 + cooling_pulse_2
    t8 = t7 + wait_time_4
    cycle_time = t8

    r = np.random.rand()  # jump threshold

    for step in range(n_steps):

        if step == 0 or ((step+1) % (n_steps//10) == 0):
            print(f"Step {step+1}/{n_steps}...")

        # Downsampled storage
        if (step % downsample_factor == 0):
            # es_projs[:, store_idx] = (proj_e * np.abs(psip) ** 2)[:N]
            # Compute all expectations (every step)
            norms_arr[store_idx] = np.linalg.norm(psip) ** 2
            avg_vals[:, store_idx] = compute_expectations(psip, e_ops, mom_array)

            store_idx += 1

        if (step % downsample_factor_psi == 0):

            if return_psis:
                psi_return[:, :, store_idx_psi] = psip
            
            store_idx_psi += 1

        # Determine which Hamiltonian and collapse operators to use
        t_present = step * eps
        time_in_cycle = t_present % cycle_time

        if time_in_cycle < t1:
            ham_ops = laser_ops
            c_ops = c_ops_lasers
        elif time_in_cycle < t2:
            ham_ops = noint_ops
        elif time_in_cycle < t3:
            ham_ops = cooling_ops
            c_ops = c_ops_cooling
        elif time_in_cycle < t4:
            ham_ops = noint_ops
        elif time_in_cycle < t5:
            ham_ops = counter_ops
            c_ops = c_ops_lasers
        elif time_in_cycle < t6:
            ham_ops = noint_ops
        elif time_in_cycle < t7:
            ham_ops = cooling_ops_counter
            c_ops = c_ops_cooling
        else:
            ham_ops = noint_ops

        (U_x_nh_eps_block, U_p_nh_eps,
         H_x_nh_eigs, V_x_nh, V_inv_x_nh,
         H_p_nh_eigs) = ham_ops

        # Soft propagation
        psi_soft = perform_soft_trajectories(psip, U_x_nh_eps_block, None, None, None, U_p_nh_eps, optim=False)
        norm_sq_next = np.linalg.norm(psi_soft) ** 2

        t_curr = 0

        while norm_sq_next < r:

            # print("SE detected: ", "normsq = ", norm_sq_next, "r = ", r)

            psi_after_jump, tstar = detect_jump_and_update(
                psip, eps - t_curr, r,
                None, # U_x_nh_eps_block not actually used during optimization
                H_x_nh_eigs, V_x_nh, V_inv_x_nh,
                H_p_nh_eigs, c_ops, hbar
            )

            r = np.random.rand()  # reset jump threshold
            t_curr += tstar # update present sub-step time

            # print("Jump detected at time = ", {step * eps + t_curr})

            jump_times.append(step * eps + t_curr)

            psi_eps_pred = finish_eps_evol(
                eps - t_curr, psi_after_jump, r,
                None, # also not used here
                H_x_nh_eigs, V_x_nh, V_inv_x_nh,
                H_p_nh_eigs, hbar
            )
            norm_eps_pred = np.linalg.norm(psi_eps_pred) ** 2

            if norm_eps_pred >= r:
                psi_soft = psi_eps_pred
                norm_sq_next = norm_eps_pred
            else:
                psip = psi_after_jump

        psip = psi_soft

    # Build return tuple
    returns = [avg_vals, norms_arr, jump_times, psi_return]

    return psip, tuple(returns)
