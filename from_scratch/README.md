# From-scratch pipeline (requires dmscatter)

This directory contains the **complete computational pipeline** for the
boosted L_10 analysis, regenerating every input from the nuclear structure
level up. Use this version if you want to recompute (or replace) the xenon
response functions yourself. If you only want to reproduce the figures from
the shipped tables, use `../make_figures.py` instead.

## Pipeline overview

| step | script | input | output |
|---|---|---|---|
| 1 | `01_extract_w.py` | dmscatter + GCN5082 OBDMs | `data/w_sigma_prime_xe.npz` — W^{Σ′}(q) for ¹²⁹/¹³¹Xe |
| 2 | `02_halo_spectra.py` | dmscatter | `data/lz_l10_halo_spectra.npz` (SHM L_10^s, m_χ = 50/200/1000 GeV) + `output/halo_validation.pdf` (note Fig. 2) |
| 3 | `03_calibrate_kcal.py` | outputs of 1+2 | `data/k_cal.txt` — empirical normalization of the manual pipeline (expect K_CAL ≈ 3.91) |
| 4 | `04_boosted_figures.py` | outputs of 1–3 | note Figures 4, 5, 6 in `output/` |

Steps 1–2 need a local **dmscatter** build; steps 3–4 are pure
numpy/scipy/matplotlib. Step 2 also validates the installation against the
DMFormFactor reference spectrum shipped with dmscatter (expect median ratio
≈ 1.08) and produces the LZ halo-spectrum comparison figure (note Fig. 2)
using the vector-extracted LZ curves shipped in `../data/lz_fig1_digitized.npz`.

## Building dmscatter

```bash
git clone https://github.com/ogorton/dmscatter.git
cd dmscatter
```

Two known issues with the current master:

1. **Compile error with recent gfortran** — an invalid format string in
   `src/parameters.f90` (~line 90). Change
   `write(6,"(I2,x2e11.4,e11.4,e11.4,e11.4)")` to
   `write(6,"(I2,x,2e11.4,e11.4,e11.4,e11.4)")` (one missing comma after `x`).
2. Then just `make` (only a Fortran compiler is needed; gfortran from
   conda-forge works: `conda install -c conda-forge gfortran`).

The xenon one-body density matrices (`targets/Xe/xe{129,131}gcn.dres`,
GCN5082 interaction) ship with the repository — no separate download.

## Running

```bash
cd from_scratch
python 01_extract_w.py    --dmscatter /path/to/dmscatter
python 02_halo_spectra.py --dmscatter /path/to/dmscatter
python 03_calibrate_kcal.py
python 04_boosted_figures.py            # or: ... fig4 / fig5 / fig6
```

Runtime: step 1 is ~1 min; step 2 runs six dmscatter event-rate jobs
(~10–20 min total); steps 3–4 are a few minutes of numerical integration.

## Expected outputs (cross-checks)

- Step 2 validation: median ratio vs `examples/test.xe131.dat` ≈ **1.08**.
- Step 2 halo peaks: m_χ = 50/200/1000 GeV → peak at **17 / 177 / 203 keV**,
  peak rates **0.070 / 0.057 / 0.031** /(t·yr·keV) (LZ preprint Fig. 1
  bottom: 0.042 / 0.036 / 0.021 — a uniform factor 1.3–1.5 lower, attributed
  to LZ's updated OBDMs).
- Step 3: **K_CAL = 3.91** with ≲ 2% spread between m_χ = 200 and 1000 GeV.
  If you replace the W tables or fix the suspected isospin/spin convention
  factor, K_CAL is where any change will show up.
- Step 4 (bin2 = 1 normalization):
  - fig4 (m_ψ = 1.4 GeV, v̄ = 0.1): F = 1.65×10¹¹, bin1 = 0.706, <100 keV = 0.213
  - fig5 (m_ψ = 1.0 GeV, v̄ = 0.14): F = 1.66×10¹¹, bin1 = 0.703, <100 keV = 0.212
  - fig6 (m_ψ = 200 GeV, v̄ = 0.03): F = 2.95×10¹⁴, bin1 = 0.695, <100 keV = 0.209;
    halo same mass at unity coupling: bin2 = 4.01 events, and at bin2 = 1:
    bin1 = 1.52, <100 keV = 1.57

## Replacing the nuclear input

To use the exact OBDMs of the LZ analysis (modified DMFormFactor-v6),
regenerate `data/w_sigma_prime_xe.npz` (arrays `q` [GeV], `W129`, `W131`)
with them via step 1, then rerun steps 2–4. Everything downstream is
unchanged.
