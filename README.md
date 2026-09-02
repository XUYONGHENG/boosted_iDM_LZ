# Boosted magnetic-moment dark matter at LZ -- figure package

Standalone code reproducing **Figures 4, 5 and 6** of the internal note
*"Boosted magnetic-moment dark matter as an interpretation of the LZ
high-energy nuclear recoil event"* (`notes/lz_boosted_md_note.pdf` in the
LZ-excess repository).

## Physics in one paragraph

The LZ event LZ230616 (E_R = 248 ± 23 ± 23 keV, 2.84 t·yr, 2.6σ global) is
interpreted with the same operator LZ itself uses -- the covariant
magnetic-moment operator L_10 (MM-MM, purely transverse spin response
Σ′, dσ/dE_R ∝ E_R² W^{Σ′}(q)) -- but with the incoming dark matter flux
**boosted** by an on-shell mediator cascade χχ → φφ → 4ψ, which produces a
narrow box-shaped velocity spectrum (Liang et al. 2021, CPC 45, 013114).
The boost moves the recoil endpoint as v², so an O(GeV) DM candidate can
fit the 248 keV event with a clean sub-100 keV region and no inelastic
splitting. Only the odd-A isotopes ¹²⁹Xe and ¹³¹Xe contribute.

## Quick start

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib
python make_figures.py              # makes all three figures in output/
python make_figures.py fig4         # or just one of fig4 / fig5 / fig6
```

Runtime is about a minute (the velocity integrals are the slow part).
The script also prints the bin-by-bin event counts (bin1 = [160,200] keV,
bin2 = [225,271] keV, low = [5.4,100] keV) with everything normalized to
one event in bin2.

## What each figure shows

| figure | output file | content |
|---|---|---|
| Fig. 4 | `output/fig4_benchmark_spectrum.pdf` | central benchmark: m_ψ = 1.4 GeV, v̄ = 0.1 (ε = 0.1); cascade m_φ = 2.80014, m_χ = 2.81424 GeV; endpoint 380 keV |
| Fig. 5 | `output/fig5_natural_benchmark.pdf` | round benchmark: m_ψ = 1.0 GeV, v̄ = 0.14; m_φ = 2.00020, m_χ = 2.02012 GeV; endpoint 382 keV |
| Fig. 6 | `output/fig6_heavy200_vs_halo.pdf` | heavy case: m_ψ = 200 GeV boosted (v̄ = 0.03, endpoint 103 MeV) vs standard-halo same mass; halo peaks inside the empty bin1, boosted peaks in bin2 |

Expected printed numbers (bin2 = 1 normalization):

```
fig4: F = 1.65e+11;  bin1 = 0.71,  <100 keV = 0.21
fig5: F = 1.66e+11;  bin1 = 0.70,  <100 keV = 0.21
fig6: boosted F = 2.95e+14, bin1 = 0.70, <100 keV = 0.21
      halo (unity coupling): bin2 = 4.01 events; at bin2=1: bin1 = 1.52,
      <100 keV = 1.57
```

## Data files (shipped, no external codes needed)

The shipped tables let you reproduce the figures without any nuclear-physics
code. If instead you want to **recompute everything from the nuclear
structure level up** -- including the W^{Σ′} tables, the LZ halo validation
spectra and the K_CAL calibration -- see `from_scratch/` (requires a local
[dmscatter](https://github.com/ogorton/dmscatter) build; build instructions
and the one-character source patch needed for recent gfortran are in
`from_scratch/README.md`).

- `data/w_sigma_prime_xe.npz` — the nuclear transverse spin responses
  W^{Σ′}(q) of ¹²⁹Xe and ¹³¹Xe on a q grid (0.005–0.8 GeV). Extracted with
  [dmscatter](https://github.com/ogorton/dmscatter) using the GCN5082
  shell-model one-body density matrices. The extraction was validated
  against a DMFormFactor reference spectrum (median ratio 1.08) and,
  end-to-end, against Fig. 1 (bottom panel) of the LZ high-energy preprint
  (shape exact, normalization a uniform factor 1.3–1.5 high, attributed to
  LZ's updated OBDMs).
- `data/halo_l10_m200.npz` — standard-halo (SHM: v0 = 238 km/s,
  v_esc = 544 km/s, v_E = 250.4 km/s, ρ₀ = 0.3 GeV/cm³) L_10^s spectrum for
  m_χ = 200 GeV at unity coupling d_10 = 1/m_v², in events/(t·yr·keV), from
  the dmscatter-based LZ-validated pipeline. Used only for the dashed halo
  curve of Fig. 6.
- `data/k_cal.txt` — empirical overall normalization K_CAL = 3.91 of the
  hand-built velocity-integral pipeline, pinned to the LZ-validated halo
  calculation (the ratio is flat in E_R and m_χ at the few-% level; a
  suspected isospin/spin-averaging convention factor, still under
  investigation). **All shape results and bin ratios are independent of
  K_CAL**; it only enters absolute normalizations (the F factors).
- `data/lz_fig1_digitized.npz` — the LZ preprint Fig. 1 (bottom) theory
  curves, extracted from the PDF vector layer (exact). Used only by the
  from-scratch halo validation plot.

## Where your help is needed (Xe response)

The only nuclear input is W^{Σ′}(q) for the two odd-A isotopes. If you have
access to the exact response tables / OBDM version used in the LZ analysis
(the modified DMFormFactor-v6), dropping them in as a replacement for
`data/w_sigma_prime_xe.npz` (same format: arrays `q` [GeV], `W129`, `W131`)
re-runs everything unchanged. This should also resolve the residual
uniform 1.3–1.5 normalization offset against the LZ halo curves, and is
the cleanest way to cross-check K_CAL.

## Caveats

- Non-relativistic EFT at leading order; at v̄ ~ 0.1–0.3 expect
  O(v²) ≲ 10% corrections (a fully relativistic MM-MM amplitude is in
  progress).
- No Earth attenuation applied; at the fitted couplings
  σ_ψXe ≳ 10⁻³² cm² this is the largest pending correction.
- Constant NR efficiency 0.96 above 5.4 keV; exposure 2.84 t·yr.

## Contact

LZ-excess project internal note, September 2026. Questions on the code or
the nuclear inputs are very welcome.
