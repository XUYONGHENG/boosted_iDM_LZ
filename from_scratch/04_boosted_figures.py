#!/usr/bin/env python3
"""04_boosted_figures.py -- note Figures 4, 5, 6 from the from-scratch data.

Boosted event rate:

    dR/dE_R = sum_T N_T int dv  Phi_tot f_box(v)  d sigma_T/dE_R ,
    Phi_tot = 4 <sigma v> J / (8 pi m_chi^2)      (Liang2021 Eq. 10)

Everything is computed at the reference normalization (d = 1/m_v^2,
<sigma v> = 3e-26 cm^3/s) and multiplied by K_CAL; the physics normalization
is then fixed by requiring exactly one event in bin2 = [225, 271] keV (the
+-1 sigma window of LZ230616), and the shape prediction is quoted as the
expected counts in bin1 = [160, 200] keV and below 100 keV.

Usage:
    python 04_boosted_figures.py [fig4|fig5|fig6]     (default: all)
Input:  data/w_sigma_prime_xe.npz    (from 01)
        data/k_cal.txt               (from 03)
        data/lz_l10_halo_spectra.npz (from 02; only the m200 curve is used)
Output: output/fig4_benchmark_spectrum.pdf, output/fig5_natural_benchmark.pdf,
        output/fig6_heavy200_vs_halo.pdf
"""
import os
import sys

import numpy as np

import l10_core as lc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

W = lc.load_W(os.path.join(DATA, "w_sigma_prime_xe.npz"))
K_CAL = float(open(os.path.join(DATA, "k_cal.txt")).read())


def model_rate(m_psi, vbar, eps=0.1):
    """Reference-normalization boosted rate on a standard grid (K_CAL applied)."""
    Qg = np.linspace(lc.ROI_KEV[0], 300., 400)
    dR, info = lc.rate_boost_L10(W, m_psi, vbar, Qg, d=1.0 / lc.MV**2, eps=eps)
    dR = dR * K_CAL
    info.update(Q_keV=Qg, dR=dR, counts=lc.bin_counts(dR, Qg))
    return info


def normalize_to_bin2(res):
    """Rate scales as <sigma v> * d^2; find the factor F with bin2 = 1."""
    b1, b2, low = res["counts"]
    F = 1. / b2 if b2 > 0 else np.inf
    return F, b1 * F, low * F


def _new_ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_yscale("log")
    ax.set_xlim(5, 400)
    for b in (lc.BIN1_KEV, lc.BIN2_KEV):
        ax.axvspan(*b, alpha=0.12)
    ax.axvline(248, ls=":", color="k", lw=1)            # LZ230616
    ax.set_xlabel(r"$E_R$ [keV]")
    ax.set_ylabel(r"d$R$/d$E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]")
    return fig, ax


def _report(tag, m_psi, vbar, res, F, b1, low):
    mT = 131 * lc.A_U
    mu = m_psi * mT / (m_psi + mT)
    Emax = 2 * mu**2 * res["v_box"][1]**2 / mT * 1e6   # endpoint at v_+ [keV]
    print("%s: m_psi=%.2f GeV, vbar=%.3f, eps=0.1" % (tag, m_psi, vbar))
    print("  x=%.5f y=%.5f -> m_phi=%.5f GeV, m_chi=%.5f GeV"
          % (res["x"], res["y"], res["m_phi"], res["m_chi"]))
    print("  endpoint at v_+: %.0f keV; Phi_tot(ref) = %.3e cm^-2 s^-1"
          % (Emax, res["phi_tot"]))
    print("  norm bin2=1: F=(<sv>/3e-26)(d m_v^2)^2 = %.3e; bin1=%.3f, "
          "<100keV=%.3f" % (F, b1, low))
    return Emax


def single_spectrum_fig(fname, title2, m_psi, vbar):
    res = model_rate(m_psi, vbar)
    F, b1, low = normalize_to_bin2(res)
    Emax = _report(fname, m_psi, vbar, res, F, b1, low)
    fig, ax = _new_ax()
    ax.plot(res["Q_keV"], res["dR"] * F, lw=1.8)
    ax.axvline(Emax, ls="--", color="gray", lw=1)
    ax.set_title(title2 + "\n" +
                 r"cascade: $x$=%.3f, $y$=%.4f $\to$ $m_\phi$=%.5f, "
                 r"$m_\chi$=%.5f GeV"
                 % (res["x"], res["y"], res["m_phi"], res["m_chi"]))
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out)
    print("  saved", out)


def fig4():
    """Note Fig. 4: central benchmark, m_psi = 1.4 GeV, vbar = 0.1."""
    single_spectrum_fig(
        "fig4_benchmark_spectrum.pdf",
        r"boosted $\mathcal{L}_{10}^s$: $m_\psi$=1.4 GeV, $\bar v$=0.1 "
        r"($\epsilon$=0.1)", 1.4, 0.1)


def fig5():
    """Note Fig. 5: round benchmark, m_psi = 1.0 GeV, vbar = 0.14."""
    single_spectrum_fig(
        "fig5_natural_benchmark.pdf",
        r"boosted $\mathcal{L}_{10}^s$: $m_\psi$=1 GeV, $\bar v$=0.14 "
        r"($\epsilon$=0.1)", 1.0, 0.14)


def fig6():
    """Note Fig. 6: heavy case -- boosted vs standard halo, same mass
    m_psi = 200 GeV, both normalized to bin2 = 1."""
    m_psi, vbar = 200.0, 0.03
    res = model_rate(m_psi, vbar)
    F, b1, low = normalize_to_bin2(res)
    _report("fig6 (boosted)", m_psi, vbar, res, F, b1, low)

    # standard-halo spectrum at the same mass (dmscatter result from 02)
    halo = np.load(os.path.join(DATA, "lz_l10_halo_spectra.npz"))
    er_h, rate_h = halo["m200"]                        # /tonne/year/keV
    dR_h = rate_h / (1e3 * lc.DAYS_PER_YR)             # /kg/day/keV
    hb1, hb2, hlow = lc.bin_counts(dR_h, er_h)
    fh = 1. / hb2
    print("fig6 (halo, unity d): bin2=%.3e at d=1/m_v^2; at bin2=1: "
          "bin1=%.3f, <100keV=%.3f" % (hb2, hb1 * fh, hlow * fh))

    fig, ax = _new_ax()
    ax.set_ylim(3e-11, 3e-7)
    ax.plot(res["Q_keV"], res["dR"] * F, lw=1.8,
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
