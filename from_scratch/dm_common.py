"""Shared helpers for driving a local dmscatter build.

The dmscatter python bindings (dmscatter/python/dmscatter.py) write a control
file and call the Fortran executable.  Two practical pitfalls, both handled
here:

1. The Fortran code reads the control-file NAME from stdin into a
   character(22) variable: long paths are silently truncated.  Always run
   from a scratch working directory with a short control-file name.
2. With recent gfortran versions, the stock source fails to compile due to
   an invalid format string in src/parameters.f90 (~line 90):
       write(6,"(I2,x2e11.4,...)")   ->   write(6,"(I2,x,2e11.4,...)")
   (one missing comma).  Apply this one-character fix before `make`.
"""
import os
import sys
import tempfile

import numpy as np

# nucleon mass in GeV (dmscatter src/constants.f90)
MN = 0.938272

# odd-A xenon isotopes: natural mass fractions
ISOTOPES = {129: 0.26401, 131: 0.21232}

# conversion: events/(kg day GeV) -> events/(tonne year keV)
KGDAY_TO_TONNEYEAR_KEV = 1000.0 * 365.25 * 1e-6

# dmscatter NucFormFactor response-function ordering (0-based index into the
# first axis of Wfunc(q)): 1:M 2:Phi'' 3:Phi~' 4:Delta 5:Sigma' 6:Sigma''
# 7:Phi''M 8:Delta Sigma'   -->  Sigma' is python index 4
OP_SIGMA_PRIME = 4


def load_dm(dmscatter_dir):
    """Return (python module, executable path) for a dmscatter checkout."""
    dmscatter_dir = os.path.abspath(dmscatter_dir)
    sys.path.append(os.path.join(dmscatter_dir, "python"))
    import dmscatter as dm
    exe = os.path.join(dmscatter_dir, "bin", "dmscatter")
    if not os.path.exists(exe):
        raise SystemExit("dmscatter executable not found at %s -- run `make` "
                         "in the dmscatter directory first (and see the "
                         "parameters.f90 note in this file's docstring)." % exe)
    return dm, exe


def in_scratch_dir(func, *args, **kwargs):
    """Run `func` inside a fresh temporary working directory (pitfall 1)."""
    prev = os.getcwd()
    with tempfile.TemporaryDirectory() as wd:
        os.chdir(wd)
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(prev)
