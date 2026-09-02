#!/usr/bin/env python3
"""03_calibrate_kcal.py -- empirical normalization of the manual pipeline.

The hand-built velocity-integral pipeline (l10_core.rate_shm_L10) reproduces
the SHAPE of the dmscatter-based, LZ-validated halo spectra exactly, but
differs by a constant, v-independent factor.  This script measures that
factor,

    K_CAL = < rate_dmscatter / rate_manual >

at m_chi = 200 and 1000 GeV (they agree at the few-% level) and writes it to
data/k_cal.txt.  All boosted absolute normalizations are multiplied by
K_CAL; shape results (bin ratios) never depend on it.

Suspected origin: an isospin/spin-averaging convention factor (2x2 = 4) in
the manual pipeline relative to dmscatter's transition_probability.  Under
investigation; flagged in the note.

Usage:
    python 03_calibrate_kcal.py
Input:  data/w_sigma_prime_xe.npz   (from 01)
        data/lz_l10_halo_spectra.npz (from 02)
Output: data/k_cal.txt
"""
import os

import numpy as np

import l10_core as lc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    W = lc.load_W(os.path.join(DATA, "w_sigma_prime_xe.npz"))
    ref = np.load(os.path.join(DATA, "lz_l10_halo_spectra.npz"))
    ks = []
    E = np.linspace(5.0, 300.0, 60)
    for mchi in [200.0, 1000.0]:
        er_ref, rate_ref = ref["m%g" % mchi]          # /tonne/year/keV
        rate_ref_dru = rate_ref / (1e3 * lc.DAYS_PER_YR)   # /kg/day/keV
        mine = lc.rate_shm_L10(W, mchi, E, d=1.0 / lc.MV**2)
        ref_i = np.interp(E, er_ref, rate_ref_dru)
        m = ref_i > 1e-3 * ref_i.max()
        ratio = ref_i[m] / mine[m]
        print("m_chi=%5.0f GeV: median ratio %.3f (p16 %.3f, p84 %.3f)"
              % (mchi, np.median(ratio), np.percentile(ratio, 16),
                 np.percentile(ratio, 84)))
        ks.append(np.median(ratio))
    K = float(np.mean(ks))
    print("K_CAL = %.3f (spread %.1f%%)" % (K, 100 * np.ptp(ks) / K))
    out = os.path.join(DATA, "k_cal.txt")
    open(out, "w").write("%g" % K)
    print("saved", out)


if __name__ == "__main__":
    main()
