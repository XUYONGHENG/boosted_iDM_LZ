#!/usr/bin/env python3
"""02_halo_spectra.py -- standard-halo L_10^s spectra from dmscatter.

Two jobs:

1. Validate the dmscatter installation against the DMFormFactor reference
   spectrum shipped with dmscatter (examples/test.xe131.dat); expect a
   median ratio ~1.08.
2. Compute the standard-halo-model L_10^s recoil spectra for
   m_chi = 50, 200, 1000 GeV at unity coupling (d_10 = 1/m_v^2) with the LZ
   halo conventions (SHM: v0 = 238 km/s, v_esc = 544 km/s, v_E = 250.5 km/s,
   rho0 = 0.3 GeV/cm^3).

dmscatter accepts only CONSTANT NREFT coefficients, while the L_10 reduction
has c_4^s(q) = 4 d q^2/m_N^2.  We use the exact algebraic decomposition

    S_A(E_R) = rate(c4^s  = 1) = C * (1/16) [W_Sig' + W_Sig'']
    S_B(E_R) = rate(c10^s = 1) = C * (q^2/4 m_N^2) W_Sig''
 =>  dR/dE_R|_{L_10} = 16 d^2 (q^2/m_N^2)^2 [ S_A - (m_N^2/4 q^2) S_B ]

where the rate kernel C cancels pointwise in E_R.  Only the odd-A isotopes
129/131 Xe contribute (transverse spin response).

Usage:
    python 02_halo_spectra.py --dmscatter /path/to/dmscatter [--skip-validation]
Outputs:
    data/lz_l10_halo_spectra.npz   (keys m50/m200/m1000, each a 2xN array
                                    [E_keV, rate / (tonne year keV)])
    output/halo_validation.pdf     (note Fig. 2: this work vs the LZ preprint
                                    curves, vector-extracted, shipped in
                                    ../data/lz_fig1_digitized.npz)
"""
import argparse
import os

import numpy as np

from dm_common import (ISOTOPES, KGDAY_TO_TONNEYEAR_KEV, MN, in_scratch_dir,
                       load_dm)
from l10_core import V0_KMS, VESC_KMS, VEARTH_KMS, RHO0

HERE = os.path.dirname(os.path.abspath(__file__))
ER_GRID = (0.5, 350.0, 0.5)      # keV: min, max, step


def run_spectrum(dm, exe, dm_dir, mchi, isotope, op_index, er_grid=ER_GRID):
    """dmscatter EventrateSpectra for a single NREFT operator (isoscalar,
    dimensionless c = 1 <=> c = 1/m_v^2 with m_v = 246.2 GeV)."""
    cs = np.zeros(15)
    cs[op_index] = 1.0
    cwords = {"wimpmass": mchi, "maxwellv0": V0_KMS, "vescape": VESC_KMS,
              "vearth": VEARTH_KMS, "dmdens": RHO0}

    def go():
        return dm.EventrateSpectra(
            Z=54, N=isotope - 54,
            dres=os.path.join(dm_dir, "targets", "Xe", "xe%sgcn" % isotope),
            controlwords=cwords,
            epmin=er_grid[0], epmax=er_grid[1], epstep=er_grid[2],
            cs=cs, exec_path=exe, name="run")
    return in_scratch_dir(go)


def l10_spectrum(dm, exe, dm_dir, mchi, d=1.0):
    """LZ L_10^s halo spectrum [/tonne/year/keV]; dimensionless d (1 = 1/m_v^2)."""
    total, er_out = None, None
    for iso, abund in ISOTOPES.items():
        mT = iso * 0.9315                            # GeV, approx nuclear mass
        er, sA = run_spectrum(dm, exe, dm_dir, mchi, iso, 3)   # c4^s = 1
        _, sB = run_spectrum(dm, exe, dm_dir, mchi, iso, 9)    # c10^s = 1
        q2 = 2.0 * mT * er * 1e-6                    # GeV^2
        sig_prime = sA - (MN**2 / (4.0 * q2)) * sB   # transverse part of c4
        if np.any(sig_prime < -0.05 * np.maximum(sA, 1e-300)):
            print("WARNING: Sigma' subtraction negative for A=%d" % iso)
        sig_prime = np.clip(sig_prime, 0.0, None)
        rate = 16.0 * d**2 * (q2 / MN**2)**2 * sig_prime
        total = rate * abund if total is None else total + rate * abund
        er_out = er
    return er_out, total * KGDAY_TO_TONNEYEAR_KEV


def validate_install(dm, exe, dm_dir):
    """Reproduce examples/test.xe131.dat (DMFormFactor reference)."""
    cs = np.zeros(15)
    cp = np.full(15, 4.8e-4)
    cn = np.full(15, 4.8e-4)

    def go():
        return dm.EventrateSpectra(
            Z=54, N=77,
            dres=os.path.join(dm_dir, "targets", "Xe", "xe131gcn"),
            controlwords={"wimpmass": 150.0, "ntscale": 2500.0},
            epmin=1, epmax=1000.0, epstep=1.0,
            cs=cs, cp=cp, cn=cn, exec_path=exe, name="val")
    er, rate = in_scratch_dir(go)
    er_ref, rate_ref = np.loadtxt(
        os.path.join(dm_dir, "examples", "test.xe131.dat"), unpack=True)
    rate_interp = np.interp(er_ref, er, rate)
    good = (rate_ref > 0) & (rate_interp > 0)
    ratio = rate_interp[good] / rate_ref[good]
    print("validation vs DMFormFactor ref: median ratio %.3f (p16 %.3f, "
          "p84 %.3f) -- expect ~1.08"
          % (np.median(ratio), np.percentile(ratio, 16),
             np.percentile(ratio, 84)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmscatter", required=True)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data",
                                                  "lz_l10_halo_spectra.npz"))
    args = ap.parse_args()
    dm_dir = os.path.abspath(args.dmscatter)
    dm, exe = load_dm(dm_dir)

    if not args.skip_validation:
        validate_install(dm, exe, dm_dir)

    out = {}
    for mchi in [50.0, 200.0, 1000.0]:
        er, rate = l10_spectrum(dm, exe, dm_dir, mchi, d=1.0)
        out[mchi] = (er, rate)
        print("m_chi=%6.0f GeV: peak at %6.1f keV, dR/dE(peak) = %.3e "
              "/tonne/yr/keV" % (mchi, er[np.argmax(rate)], rate.max()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **{"m%g" % m: np.vstack(v) for m, v in out.items()})
    print("saved", os.path.abspath(args.out))

    # validation plot vs the vector-extracted LZ preprint curves
    dig_path = os.path.join(HERE, "..", "data", "lz_fig1_digitized.npz")
    if os.path.exists(dig_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dig = np.load(dig_path)
        colors = {50.0: "#d62728", 200.0: "#9467bd", 1000.0: "#8c564b"}
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for mchi, (er, rate) in out.items():
            ax.plot(er, rate, color=colors[mchi],
                    label=r"this work, %g GeV/$c^2$" % mchi)
            E, r = dig["m%g" % mchi]
            ax.plot(E, r, ls="--", lw=1.4, color=colors[mchi],
                    label=r"LZ preprint, %g GeV/$c^2$" % mchi)
        ax.axvspan(0, 5.4, color="gray", alpha=0.3, lw=0)
        ax.axvspan(269.9, 350, color="gray", alpha=0.3, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(0, 350)
        ax.set_ylim(1e-3, 3e-1)
        ax.set_xlabel("True Recoil Energy [keV]")
        ax.set_ylabel(r"Differential Rate [/tonne/year/keV]")
        ax.legend(fontsize=8, ncol=2,
                  title=r"$\mathcal{L}_{10}^s$, $d_{10}=1/m_v^2$")
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
        outpdf = os.path.join(HERE, "output", "halo_validation.pdf")
        fig.savefig(outpdf)
        print("saved", outpdf)


if __name__ == "__main__":
    main()
