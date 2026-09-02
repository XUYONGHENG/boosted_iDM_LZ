#!/usr/bin/env python3
"""
make_figures.py -- boosted magnetic-moment (L_10) dark matter at LZ
===================================================================

Self-contained script reproducing Figures 4, 5 and 6 of the internal note
"Boosted magnetic-moment dark matter as an interpretation of the LZ
high-energy nuclear recoil event" (lz_boosted_md_note.pdf).

Physics summary
---------------
Detector interaction: the LZ covariant magnetic-moment operator

    L_10 = d_10 [psibar i sigma^{mu nu} (q_nu/m_N) psi]
                [Nbar   i sigma_{mu alpha} (q^alpha/m_N) N]

whose non-relativistic reduction is a purely transverse spin-spin coupling,

    R^{Sigma'} = d_10^2 (q^2/m_N^2)^2 ,   R^{Sigma''} = 0   (j_psi = 1/2)

so that  d sigma_T / dE_R = (2 m_T / (v^2 (2 J_T + 1)))  R^{Sigma'}
W_T^{Sigma'}(q),  with q^2 = 2 m_T E_R.  Only the odd-A xenon isotopes
129Xe (J=1/2) and 131Xe (J=3/2) contribute.

Boost: hidden-sector on-shell cascade  chi chi -> phi phi -> 4 psi
(Liang et al. 2021, CPC 45, 013114), giving a narrow box-shaped velocity
spectrum f(v) on [v_-, v_+] with edges  v_pm = (x +/- y)/(1 +/- x y).

Event rate per unit detector mass:

    dR/dE_R = sum_T N_T int dv  Phi_tot f(v)  d sigma_T / dE_R ,
    Phi_tot = 4 <sigma v> J / (8 pi m_chi^2)      (Liang2021 Eq. 10)

Data files (no external dependencies beyond numpy/scipy/matplotlib)
-------------------------------------------------------------------
data/w_sigma_prime_xe.npz : W^{Sigma'}(q) tables for 129/131 Xe, extracted
    from dmscatter (https://github.com/ogorton/dmscatter, GCN5082 shell-model
    one-body density matrices); the extraction was validated against a
    DMFormFactor reference spectrum (median ratio 1.08) and, end-to-end,
    against Fig. 1 (bottom) of the LZ high-energy preprint.
data/halo_l10_m200.npz    : standard-halo (SHM) L_10^s spectrum for
    m_chi = 200 GeV at unity coupling d_10 = 1/m_v^2 (LZ halo conventions:
    v0 = 238 km/s, v_esc = 544 km/s, v_E = 250.4 km/s, rho0 = 0.3 GeV/cm^3),
    in events / (tonne year keV), from the dmscatter-based, LZ-validated
    pipeline.  Used for the halo curve of Fig. 6.
data/k_cal.txt            : empirical overall normalization K_CAL = 3.91
    of the hand-built velocity-integral pipeline, pinned to the
    LZ-validated halo calculation (flat in E_R and m_chi at the few-%
    level; suspected isospin/spin-averaging convention factor, under
    investigation).  All shape results are independent of K_CAL.

Usage
-----
    python make_figures.py            # produces all three figures
    python make_figures.py fig4       # only one of fig4 / fig5 / fig6

Outputs are written to output/ and the bin-by-bin event counts are printed.
"""
import os
import sys
from math import erf

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ constants
MN = 0.938272           # GeV, nucleon mass (m_M = m_N convention of L_10)
MV = 246.2            # GeV, weak scale (LZ unity coupling: d_10 = 1/MV^2)
GEV2_TO_CM2 = 3.89379338e-28
KG_PER_GEV = 1.78266192e-27     # kg per GeV/c^2
SEC_PER_DAY = 86400.
DAYS_PER_YR = 365.25
A_U = 0.9315                    # GeV per atomic mass unit

# odd-A xenon isotopes: natural mass fraction xi and nuclear spin J
XE = {129: dict(xi=0.26401, J=0.5), 131: dict(xi=0.21232, J=1.5)}

# LZ high-energy analysis (2026 preprint)
EXPOSURE_TYR = 2.84                     # tonne year
ROI_KEV = (5.4, 270.)                   # region of interest
BIN1_KEV = (160., 200.)                 # empty bin below the event
BIN2_KEV = (225., 271.)                 # +-1 sigma window of LZ230616 (248 keV)
LOW_KEV = (5.4, 100.)                   # background-dominated low-energy region

# annihilation-boost flux normalization (Liang2021 Eq. 10)
J_NFW = 1.0e23                          # GeV^2/cm^5, NFW gamma=1
SIGMAV_REF = 3.0e-26                    # cm^3/s, thermal reference

# empirical normalization of the hand-built pipeline (see module docstring)
K_CAL = float(open(os.path.join(DATA, "k_cal.txt")).read())

# W^{Sigma'} tables (q in GeV)
_W = np.load(os.path.join(DATA, "w_sigma_prime_xe.npz"))


def W_sigma_prime(iso, q):
    """Nuclear transverse spin response W^{Sigma'}_{00}(q) of isotope `iso`,
    linearly interpolated on the shipped table; q in GeV."""
    return np.interp(q, _W["q"], _W["W%d" % iso], left=0., right=0.)


# --------------------------------------------------------------- cross section
def dsig_dER_L10(v, EkeV, iso, d=1.0):
    """Differential cross section d sigma_T / dE_R for L_10 at DM speed v
    (natural units, c = 1) and recoil energy EkeV [keV], in GeV^-3.
    d is the Wilson coefficient in GeV^-2 (d = 1/MV^2 is the LZ unity
    coupling)."""
    mT = iso * A_U
    J = XE[iso]["J"]
    E = EkeV * 1e-6                       # GeV
    q2 = 2.0 * mT * E
    R = d**2 * (q2 / MN**2)**2            # R^{Sigma'}, j_psi = 1/2
    return (2.0 * mT / (v**2 * (2.0 * J + 1.0))) * R * W_sigma_prime(
        iso, np.sqrt(q2))


# ------------------------------------------------------------- box spectrum
def solve_xy(vbar, eps=0.1):
    """Invert v_pm = vbar (1 +/- eps) to the cascade boosts (x, y)."""
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
    m_chi = m_phi / np.sqrt(1 - x**2)
    return m_chi, m_phi


def box_vmin(x, y):
    return abs(x - y) / (1 - x * y)


def box_vmax(x, y):
    return (x + y) / (1 + x * y)


def box_fv(x, y, v):
    """Normalized box velocity pdf f(v) (Liang2021 Eq. 2), v in units of c."""
    v = np.asarray(v, dtype=float)
    pre = np.sqrt((1 - x**2) * (1 - y**2)) / (2 * x * y)
    f = pre * v / (1 - v**2)**1.5
    return np.where((v >= box_vmin(x, y)) & (v <= box_vmax(x, y)), f, 0.)


# ------------------------------------------------------------------ rate
def rate_boost_L10(m_psi, vbar, EkeV_grid, d=1.0, eps=0.1,
                   sigmav=SIGMAV_REF, J=J_NFW, n_v=600):
    """Boosted L_10 differential rate [keV^-1 kg^-1 day^-1] at the reference
    flux (sigmav) and coupling (d).  Returns (rate, info-dict)."""
    x, y = solve_xy(vbar, eps)
    m_chi, m_phi = cascade_masses(m_psi, x, y)
    phi_tot = 4 * sigmav * J / (8 * np.pi * m_chi**2)   # cm^-2 s^-1
    vm, vp = box_vmin(x, y), box_vmax(x, y)
    out = np.zeros_like(EkeV_grid, dtype=float)
    for iso, prm in XE.items():
        mT = iso * A_U
        Nt = prm["xi"] / (mT * KG_PER_GEV)      # target nuclei per kg nat. Xe
        mu = m_psi * mT / (m_psi + mT)
        for i, E in enumerate(EkeV_grid):
            vmin = np.sqrt(mT * E * 1e-6 / 2.0) / mu  # elastic v_min
            if vmin >= vp:
                continue
            v = np.linspace(max(vmin, vm * (1 + 1e-9)), vp * (1 - 1e-9), n_v)
            fv = phi_tot * box_fv(x, y, v)            # cm^-2 s^-1 per dv
            ds = dsig_dER_L10(v, E, iso, d)           # GeV^-3
            integ = np.trapezoid(fv * ds, v)
            out[i] += Nt * integ * GEV2_TO_CM2 * 1e-6 * SEC_PER_DAY
    return out, dict(x=x, y=y, m_chi=m_chi, m_phi=m_phi, phi_tot=phi_tot,
                     v_box=(vm, vp))


# ------------------------------------------------------- LZ bin bookkeeping
def eff_NR(Q_keV):
    """Constant NR efficiency 0.96 above the 5.4 keV ROI threshold."""
    return np.where(np.asarray(Q_keV, dtype=float) >= ROI_KEV[0], 0.96, 0.)


def events_in_bin(Q_keV, dR, bin_keV, exposure_tyr=EXPOSURE_TYR):
    """Expected events in [bin_keV] from a rate dR [keV^-1 kg^-1 day^-1]."""
    m = (Q_keV >= bin_keV[0]) & (Q_keV <= bin_keV[1])
    kgday = exposure_tyr * 1e3 * DAYS_PER_YR
    return np.trapezoid(dR[m] * eff_NR(Q_keV[m]), Q_keV[m]) * kgday


def bin_counts(dR, Q_keV):
    return (events_in_bin(Q_keV, dR, BIN1_KEV),
            events_in_bin(Q_keV, dR, BIN2_KEV),
            events_in_bin(Q_keV, dR, LOW_KEV))


def model_rate(m_psi, vbar, eps=0.1):
    """Reference-normalization (d = 1/MV^2, sigmav = 3e-26) boosted rate on a
    standard grid, with K_CAL applied.  Bin ratios are K_CAL-independent."""
    Qg = np.linspace(ROI_KEV[0], 300., 400)
    dR, info = rate_boost_L10(m_psi, vbar, Qg, d=1.0 / MV**2, eps=eps)
    dR = dR * K_CAL
    info.update(Q_keV=Qg, dR=dR, counts=bin_counts(dR, Qg))
    return info


def normalize_to_bin2(res):
    """The rate scales as <sigma v> * d^2; find the factor f with bin2 = 1."""
    b1, b2, low = res["counts"]
    f = 1. / b2 if b2 > 0 else np.inf
    return f, b1 * f, low * f


# --------------------------------------------------------------- figures
def _new_ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_yscale("log")
    ax.set_xlim(5, 400)
    for b in (BIN1_KEV, BIN2_KEV):
        ax.axvspan(*b, alpha=0.12)
    ax.axvline(248, ls=":", color="k", lw=1)          # LZ230616
    ax.set_xlabel(r"$E_R$ [keV]")
    ax.set_ylabel(r"d$R$/d$E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]")
    return fig, ax


def _report(tag, m_psi, vbar, res, f, b1, low):
    mT = 131 * A_U
    mu = m_psi * mT / (m_psi + mT)
    Emax = 2 * mu**2 * res["v_box"][1]**2 / mT * 1e6   # endpoint at v_+ [keV]
    print("%s: m_psi=%.2f GeV, vbar=%.3f, eps=0.1" % (tag, m_psi, vbar))
    print("  x=%.5f y=%.5f -> m_phi=%.5f GeV, m_chi=%.5f GeV"
          % (res["x"], res["y"], res["m_phi"], res["m_chi"]))
    print("  endpoint at v_+: %.0f keV; Phi_tot(ref) = %.3e cm^-2 s^-1"
          % (Emax, res["phi_tot"]))
    print("  norm bin2=1: F=(<sv>/3e-26)(d m_v^2)^2 = %.3e; bin1=%.3f, "
          "<100keV=%.3f" % (f, b1, low))
    return Emax


def fig4():
    """Note Fig. 4: central benchmark, m_psi = 1.4 GeV, vbar = 0.1."""
    res = model_rate(1.4, 0.1)
    f, b1, low = normalize_to_bin2(res)
    Emax = _report("fig4", 1.4, 0.1, res, f, b1, low)
    fig, ax = _new_ax()
    ax.plot(res["Q_keV"], res["dR"] * f, lw=1.8)
    ax.axvline(Emax, ls="--", color="gray", lw=1)       # endpoint at v_+
    ax.set_title(r"boosted $\mathcal{L}_{10}^s$: $m_\psi$=1.4 GeV, "
                 r"$\bar v$=0.1 ($\epsilon$=0.1)" + "\n" +
                 r"cascade: $x$=%.3f, $y$=%.4f $\to$ $m_\phi$=%.5f, "
                 r"$m_\chi$=%.5f GeV"
                 % (res["x"], res["y"], res["m_phi"], res["m_chi"]))
    fig.tight_layout()
    out = os.path.join(OUT, "fig4_benchmark_spectrum.pdf")
    fig.savefig(out)
    print("  saved", out)


def fig5():
    """Note Fig. 5: round benchmark, m_psi = 1.0 GeV, vbar = 0.14."""
    res = model_rate(1.0, 0.14)
    f, b1, low = normalize_to_bin2(res)
    Emax = _report("fig5", 1.0, 0.14, res, f, b1, low)
    fig, ax = _new_ax()
    ax.plot(res["Q_keV"], res["dR"] * f, lw=1.8)
    ax.axvline(Emax, ls="--", color="gray", lw=1)
    ax.set_title(r"boosted $\mathcal{L}_{10}^s$: $m_\psi$=1 GeV, "
                 r"$\bar v$=0.14 ($\epsilon$=0.1)" + "\n" +
                 r"cascade: $x$=%.3f, $y$=%.4f $\to$ $m_\phi$=%.5f, "
                 r"$m_\chi$=%.5f GeV"
                 % (res["x"], res["y"], res["m_phi"], res["m_chi"]))
    fig.tight_layout()
    out = os.path.join(OUT, "fig5_natural_benchmark.pdf")
    fig.savefig(out)
    print("  saved", out)


def fig6():
    """Note Fig. 6: heavy case -- boosted vs standard halo, same mass
    m_psi = 200 GeV, both normalized to bin2 = 1."""
    m_psi, vbar = 200.0, 0.03
    res = model_rate(m_psi, vbar)
    f, b1, low = normalize_to_bin2(res)
    Emax = _report("fig6 (boosted)", m_psi, vbar, res, f, b1, low)

    # standard-halo spectrum at the same mass (shipped, LZ-validated curve)
    halo = np.load(os.path.join(DATA, "halo_l10_m200.npz"))
    er_h = halo["E_keV"]
    dR_h = halo["dR_dE_per_t_yr_keV"] / (1e3 * DAYS_PER_YR)  # /kg/day/keV
    hb1, hb2, hlow = bin_counts(dR_h, er_h)
    fh = 1. / hb2
    print("fig6 (halo, unity d): bin2=%.3e at d=1/m_v^2; at bin2=1: "
          "bin1=%.3f, <100keV=%.3f" % (hb2, hb1 * fh, hlow * fh))

    fig, ax = _new_ax()
    ax.set_ylim(3e-11, 3e-7)
    ax.plot(res["Q_keV"], res["dR"] * f, lw=1.8,
            label=r"boosted: $\bar v$=0.03, $\epsilon$=0.1")
    ax.plot(er_h, dR_h * fh, lw=1.8, ls="--", color="darkred",
            label=r"standard halo (SHM), same $m_\psi$")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(r"boosted $\mathcal{L}_{10}^s$ vs halo: $m_\psi$=200 GeV, "
                 r"bin2=1" + "\n" +
                 r"boosted: $x$=%.3f, $y$=%.4f $\to$ $m_\phi$=%.2f, "
                 r"$m_\chi$=%.2f GeV"
                 % (res["x"], res["y"], res["m_phi"], res["m_chi"]),
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "fig6_heavy200_vs_halo.pdf")
    fig.savefig(out)
    print("  saved", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "fig4"):
        fig4()
    if which in ("all", "fig5"):
        fig5()
    if which in ("all", "fig6"):
        fig6()
