"""Physical parameters, the natural-unit conversion, and grid sizing.

Everything the simulation needs to be reproduced is here.  The atom is Be+
driven in a three-level Lambda EIT configuration on a single axial mode.

Units: hbar = m = 1, time in Gamma_1^-1, energy in hbar Gamma_1.  US_PER_NAT
converts a natural time to microseconds.
"""
import numpy as np

# ---------------------------------------------------------------------
# SI inputs
# ---------------------------------------------------------------------
hbar_SI = 1.054571817e-34
c_SI    = 2.99792458e8

m_SI      = 1.496536e-26           # kg
omega_z   = 2*np.pi*1.59e6         # rad/s, axial trap frequency
Omega1_SI = 2*np.pi*33.9e6         # Rabi frequency, leg 1
Omega2_SI = 2*np.pi*33.9e6         # Rabi frequency, leg 2
Delta_SI  = 2*np.pi*360e6          # common detuning from the excited state
Gamma1_SI = 2*np.pi*6e6            # partial width to level 0; the unit of rate
Gamma2_SI = 2*np.pi*12e6           # partial width to level 1

nu_laser  = 957e12 + 360e6         # Hz
theta_deg = 10.0                   # beam angle to the trap axis
kz_SI     = 2*np.pi*nu_laser*np.sin(np.deg2rad(theta_deg))/c_SI
kph_SI    = 2*np.pi*nu_laser/c_SI  # an emitted photon carries the full |k|

x0_SI  = np.sqrt(hbar_SI/(m_SI*omega_z))
eta    = kz_SI*x0_SI               # laser Lamb-Dicke parameter, derived
eta_ph = kph_SI*x0_SI              # recoil scale = eta/sin(theta)

# ---------------------------------------------------------------------
# Natural units
# ---------------------------------------------------------------------
hbar, m = 1, 1

Gamma_g_actual = Gamma1_SI/(2*np.pi)/1e6       # MHz
US_PER_NAT     = 1.0/(2*np.pi*Gamma_g_actual)  # Gamma_1^-1 -> us

Gamma_g   = 1.0
Gamma_e   = Gamma2_SI/Gamma1_SI                # 2
Gamma_tot = Gamma_g + Gamma_e
nu_trap   = omega_z/Gamma1_SI                  # 0.265
Delta_R   = Delta_SI/Gamma1_SI                 # 60
Omega_g   = Omega1_SI/Gamma1_SI                # 5.65
Omega_e   = Omega2_SI/Gamma1_SI                # 5.65

x0     = np.sqrt(hbar/(m*nu_trap))
p0     = np.sqrt(m*nu_trap)                    # = 1/x0
k_beam = eta/x0

n_levels  = 3
se_coeffs = [(Gamma_g, 2, 0), (Gamma_e, 2, 1)]   # (rate, from, to)

# Bright-state AC Stark shift.  The EIT condition is that this sits on nu_trap,
# so that the Fano absorption zero falls on the carrier and the bright
# resonance on the red sideband.
dAC     = (np.sqrt(Delta_R**2 + Omega_g**2 + Omega_e**2) - Delta_R)/2
Gamma_b = Gamma_tot*(Omega_g**2 + Omega_e**2)/(4*Delta_R**2 + Gamma_tot**2)

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
SAFETY    = 1.5     # margin on the classical turning point, in both x and p
KDP_FLOOR = 4.0     # minimum resolution of the laser wavenumber, k_beam/dp
N_MIN     = 256
N_MAX     = 2**18


def size_grid(n_hi, safety=SAFETY, kdp_floor=KDP_FLOOR, N_min=N_MIN, N_max=N_MAX):
    """Smallest power-of-two grid holding Fock states up to n_hi.

    With p = linspace(-pmax, pmax, N) and x = 2 pi fftfreq(N, dp),

        L_half = pi/dp,     dx = 2 pi/(N dp)

    Containing the classical turning point in both x and p, with margin
    `safety`, forces

        N >= (2/pi) safety^2 (2 n_hi + 1).

    A Fock state is band-limited by its turning momentum, so `safety` only a
    little above 1 already gives machine-precision representation: measured
    weight in the outer tenth of the box is ~1e-26 at safety = 1.25 and
    9e-2 at safety = 1.10.

    `kdp_floor` separately keeps k_beam/dp above a floor, adding

        N >= 2 safety sqrt(2 n_hi + 1) kdp_floor / eta.

    This is not tied to spontaneous emission: the lasers transfer +-k per
    photon and +-2k per Raman cycle either way, and dp grows at small n
    (pmax ~ sqrt(n) while N ~ n), so without the floor k_beam/dp falls below 1
    at the bottom of the range.  Pass 0 to disable.
    """
    turn  = np.sqrt(2*n_hi + 1)
    N_box = (2/np.pi)*safety**2*(2*n_hi + 1)
    N_rec = 2*safety*turn*kdp_floor/eta if kdp_floor > 0 else 0.0
    N = 2**int(np.ceil(np.log2(max(N_box, N_rec, N_min))))
    if N > N_max:
        raise ValueError("n_hi=%r needs N=%d > N_max=%d" % (n_hi, N, N_max))

    # N fixes the PRODUCT of the two margins, since L_half = pi(N-1)/(2 pmax)
    # and x0 p0 = 1:       s_x s_p = pi (N-1) / (2 (2 n_hi + 1))
    # The split between them is free, so choose pmax to equalise them: the
    # binding margin is the smaller of the two, and rounding N up to a power
    # of two otherwise dumps all the slack into position.
    s_bal = np.sqrt(np.pi*(N - 1)/(2*(2*n_hi + 1)))
    pmax  = s_bal*turn*p0
    if kdp_floor > 0:
        pmax = min(pmax, k_beam*(N - 1)/(2*kdp_floor))
    return N, pmax


def thermal_cap(nbar, tol=1e-3):
    """Fock cutoff retaining 1 - tol of a geometric distribution of mean nbar."""
    if nbar <= 0:
        return 1, 0.0
    q = nbar/(1.0 + nbar)
    ncap = max(int(np.ceil(np.log(tol)/np.log(q))), 1)
    return ncap, q**ncap
