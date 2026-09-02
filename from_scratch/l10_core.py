"""Core physics for the boosted L_10 pipeline (from-scratch version).

Everything here is plain numpy/scipy; dmscatter is only needed to GENERATE
the nuclear response tables (script 01), not to use them.

Conventions
-----------
* c = 1 throughout; velocities in units of c; energies in GeV unless a
  variable name carries a _KEV/_keV suffix.
* The L_10 Wilson coefficient d is in GeV^-2; d = 1/MV^2 with MV = 246.2 GeV
  is the LZ "unity coupling" convention.
* Rates are in events / (keV kg day).
"""
import os

import numpy as np

# ------------------------------------------------------------------ constants
MN = 0.938272           # GeV, nucleon mass (the m_M = m_N scale of L_10)
MV = 246.2            # GeV, weak scale (LZ unity coupling d_10 = 1/MV^2)
GEV2_TO_CM2 = 3.89379338e-28
KG_PER_GEV = 1.78266192e-27     # kg per GeV/c^2
SEC_PER_DAY = 86400.
DAYS_PER_YR = 365.25
A_U = 0.9315                    # GeV per atomic mass unit
KGDAY_TO_TONNEYEAR_KEV = 1000.0 * 365.25 * 1e-6

# odd-A xenon isotopes: natural mass fraction xi and nuclear spin J
XE = {129: dict(xi=0.26401, J=0.5), 131: dict(xi=0.21232, J=1.5)}

# LZ high-energy analysis (2026 preprint)
EXPOSURE_TYR = 2.84                     # tonne year
ROI_KEV = (5.4, 270.)                   # region of interest [keV]
BIN1_KEV = (160., 200.)                 # empty bin below the event
BIN2_KEV = (225., 271.)                 # +-1 sigma window of LZ230616 (248 keV)
LOW_KEV = (5.4, 100.)                   # background-dominated low-energy region

# standard halo model, LZ conventions (LZ preprint / PRL 2404.17666)
V0_KMS = 238.0
VESC_KMS = 544.0
VEARTH_KMS = np.sqrt(11.1**2 + (12.2 + 238.0)**2 + 7.3**2)   # ~250.5 km/s
RHO0 = 0.3                              # GeV/cm^3
C_KMS = 299792.458
C_CM_S = 2.99792458e10

# annihilation-boost flux normalization (Liang2021 Eq. 10)
J_NFW = 1.0e23                          # GeV^2/cm^5, NFW gamma = 1
SIGMAV_REF = 3.0e-26                    # cm^3/s, thermal reference


# ------------------------------------------------------------- W^{Sigma'} tables
def load_W(path):
    """Load W^{Sigma'}(q) tables written by 01_extract_w.py."""
    return np.load(path)


def W_sigma_prime(Wtab, iso, q):
    """Linear interpolation of W^{Sigma'}_{00}(q); q in GeV."""
    return np.interp(q, Wtab["q"], Wtab["W%d" % iso], left=0., right=0.)


# --------------------------------------------------------------- cross section
def dsig_dER_L10(Wtab, v, EkeV, iso, d=1.0):
    """d sigma_T/dE_R for L_10 at DM speed v (c=1), E_R in keV -> GeV^-3.

    d sigma_T/dE_R = (2 m_T / (v^2 (2 J_T + 1)))  R^{Sigma'}  W^{Sigma'}_T(q)
    with R^{Sigma'} = d^2 (q^2/m_N^2)^2 (j_psi = 1/2), q^2 = 2 m_T E_R.
    """
    mT = iso * A_U
    J = XE[iso]["J"]
    E = EkeV * 1e-6                       # GeV
    q2 = 2.0 * mT * E
    R = d**2 * (q2 / MN**2)**2
    return (2.0 * mT / (v**2 * (2.0 * J + 1.0))) * R * W_sigma_prime(
        Wtab, iso, np.sqrt(q2))


# ------------------------------------------------- standard halo model (SHM)
def shm_speed_pdf(v):
    """Earth-frame SHM speed pdf f_E(v) (Lewin & Smith), v in units of c."""
    from math import erf
    v0 = V0_KMS / C_KMS
    ve = VEARTH_KMS / C_KMS
    vesc = VESC_KMS / C_KMS
    z = vesc / v0
    Nesc = erf(z) - 2.0 * z * np.exp(-z * z) / np.sqrt(np.pi)
    x = v / v0
    xe = ve / v0
    f = np.where(
        v <= vesc + ve,
        x / (xe * np.sqrt(np.pi) * Nesc * v0) *
        (np.exp(-((x - xe)**2)) - np.exp(-(np.minimum(x + xe, z))**2)),
        0.0)
    return np.where(v > 0, f, 0.0)


def rate_shm_L10(Wtab, mchi, EkeV_grid, d=1.0, n_v=4000):
    """Manual SHM halo L_10^s rate [keV^-1 kg^-1 day^-1].

    Used for the K_CAL calibration against the dmscatter halo result."""
    out = np.zeros_like(EkeV_grid, dtype=float)
    vesc_eff = (VESC_KMS + VEARTH_KMS) / C_KMS
    for iso, prm in XE.items():
        mT = iso * A_U
        mu = mchi * mT / (mchi + mT)
        Nt = prm["xi"] / (mT * KG_PER_GEV)      # nuclei per kg nat. Xe
        for i, E in enumerate(EkeV_grid):
            vmin = np.sqrt(mT * E * 1e-6 / 2.0) / mu
            if vmin >= vesc_eff:
                continue
            v = np.linspace(vmin * (1 + 1e-9), vesc_eff, n_v)
            flux = (RHO0 / mchi) * shm_speed_pdf(v) * v * C_CM_S
            ds = dsig_dER_L10(Wtab, v, E, iso, d)
            out[i] += Nt * np.trapezoid(flux * ds, v) * GEV2_TO_CM2 \
                * 1e-6 * SEC_PER_DAY
    return out


# ------------------------------------------------------------- box spectrum
def solve_xy(vbar, eps=0.1):
    """Invert box edges v_pm = vbar (1 +/- eps) to cascade boosts (x, y)."""
    from scipy.optimize import brentq
    vp, vm = vbar * (1 + eps), vbar * (1 - eps)

    def f(yy):
        xx = (vp - yy) / (1 - vp * yy)
        return (xx - yy) / (1 - xx * yy) - vm
    y = brentq(f, 1e-14, vp * (1 - 1e-12))
    return (vp - y) / (1 - vp * y), y


def cascade_masses(m_psi, x, y):
    """On-shell cascade relations: m_phi -> psi psi, chi chi -> phi phi."""
    m_phi = 2 * m_psi / np.sqrt(1 - y**2)
    return m_phi / np.sqrt(1 - x**2), m_phi      # (m_chi, m_phi)


def box_vmin(x, y):
    return abs(x - y) / (1 - x * y)


def box_vmax(x, y):
    return (x + y) / (1 + x * y)


def box_fv(x, y, v):
    """Normalized box velocity pdf f(v) (Liang2021 Eq. 2)."""
    v = np.asarray(v, dtype=float)
    pre = np.sqrt((1 - x**2) * (1 - y**2)) / (2 * x * y)
    f = pre * v / (1 - v**2)**1.5
    return np.where((v >= box_vmin(x, y)) & (v <= box_vmax(x, y)), f, 0.)


# --------------------------------------------------------------- boosted rate
def rate_boost_L10(Wtab, m_psi, vbar, EkeV_grid, d=1.0, eps=0.1,
                   sigmav=SIGMAV_REF, J=J_NFW, n_v=600):
    """Boosted L_10 differential rate [keV^-1 kg^-1 day^-1] at reference flux
    and coupling.  Returns (rate, info-dict with cascade parameters)."""
    x, y = solve_xy(vbar, eps)
    m_chi, m_phi = cascade_masses(m_psi, x, y)
    phi_tot = 4 * sigmav * J / (8 * np.pi * m_chi**2)   # cm^-2 s^-1
    vm, vp = box_vmin(x, y), box_vmax(x, y)
    out = np.zeros_like(EkeV_grid, dtype=float)
    for iso, prm in XE.items():
        mT = iso * A_U
        Nt = prm["xi"] / (mT * KG_PER_GEV)
        mu = m_psi * mT / (m_psi + mT)
        for i, E in enumerate(EkeV_grid):
            vmin = np.sqrt(mT * E * 1e-6 / 2.0) / mu   # elastic v_min
            if vmin >= vp:
                continue
            v = np.linspace(max(vmin, vm * (1 + 1e-9)), vp * (1 - 1e-9), n_v)
            fv = phi_tot * box_fv(x, y, v)             # cm^-2 s^-1 per dv
            ds = dsig_dER_L10(Wtab, v, E, iso, d)      # GeV^-3
            out[i] += Nt * np.trapezoid(fv * ds, v) * GEV2_TO_CM2 \
                * 1e-6 * SEC_PER_DAY
    return out, dict(x=x, y=y, m_chi=m_chi, m_phi=m_phi, phi_tot=phi_tot,
                     v_box=(vm, vp))


# ------------------------------------------------------- LZ bin bookkeeping
def eff_NR(Q_keV):
    """Constant NR efficiency 0.96 above the 5.4 keV ROI threshold."""
    return np.where(np.asarray(Q_keV, dtype=float) >= ROI_KEV[0], 0.96, 0.)


def events_in_bin(Q_keV, dR, bin_keV, exposure_tyr=EXPOSURE_TYR):
    m = (Q_keV >= bin_keV[0]) & (Q_keV <= bin_keV[1])
    kgday = exposure_tyr * 1e3 * DAYS_PER_YR
    return np.trapezoid(dR[m] * eff_NR(Q_keV[m]), Q_keV[m]) * kgday


def bin_counts(dR, Q_keV):
    return (events_in_bin(Q_keV, dR, BIN1_KEV),
            events_in_bin(Q_keV, dR, BIN2_KEV),
            events_in_bin(Q_keV, dR, LOW_KEV))
