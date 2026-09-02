#!/usr/bin/env python3
"""01_extract_w.py -- extract W^{Sigma'}(q) for 129/131 Xe from dmscatter.

Runs dmscatter's NucFormFactor with the GCN5082 shell-model one-body density
matrices (shipped with dmscatter as targets/Xe/xe{129,131}gcn.dres) and saves
the transverse spin response on a momentum grid q = 0.005..0.8 GeV.

W^{Sigma'} is a purely nuclear quantity (independent of the DM mass); the
control word wimpmass is only needed by dmscatter's internal setup, and
usemomentum=1 makes the grid a momentum grid in GeV.

Usage:
    python 01_extract_w.py --dmscatter /path/to/dmscatter
Output:
    data/w_sigma_prime_xe.npz   (arrays: q [GeV], W129, W131)
"""
import argparse
import os

import numpy as np

from dm_common import ISOTOPES, OP_SIGMA_PRIME, in_scratch_dir, load_dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmscatter", required=True,
                    help="path to the dmscatter checkout (with bin/dmscatter)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data",
        "w_sigma_prime_xe.npz"))
    args = ap.parse_args()
    dm_dir = os.path.abspath(args.dmscatter)
    dm, exe = load_dm(dm_dir)

    q = np.linspace(0.005, 0.8, int(round((0.8 - 0.005) / 0.005)) + 1)
    out = {"q": q}
    for iso in ISOTOPES:
        def go():
            Wfunc = dm.NucFormFactor(
                Z=54, N=iso - 54,
                dres=os.path.join(dm_dir, "targets", "Xe", "xe%sgcn" % iso),
                controlwords={"wimpmass": 1000.0, "usemomentum": 1},
                epmin=q[0], epmax=q[-1], epstep=q[1] - q[0],
                exec_path=exe, name="wf")
            # evaluate while still inside the scratch working directory
            return np.array([Wfunc(qq)[OP_SIGMA_PRIME, 0, 0] for qq in q])
        out["W%d" % iso] = in_scratch_dir(go)
        print("isotope %d: W^Sigma' peak %.4f at q = %.3f GeV"
              % (iso, out["W%d" % iso].max(),
                 q[np.argmax(out["W%d" % iso])]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print("saved", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
