# Current limitations of the displacement/distortion products

This documents known limitations of the outputs produced by `download_nisar.py` +
`process_displacement.py` (`rate_vertical_mm_per_year.tif`,
`angular_distortion_mm_per_yr_per_m.tif`), as of the current NISAR data releases
(`NISAR_L2_GUNW_PROVISIONAL_V1`).

## Summary

The vertical displacement **rate** map is not physically reliable in absolute terms
and should be treated as a methodology proof-of-concept rather than a quantitative
science product. The **angular distortion** map is less affected in practice and,
in the current test runs, tracks real geographic/terrain features coherently — it
is considered satisfactory for the present, small-scale (few-km) proof-of-concept
scope.

## What we observed

For a test AOI in the Brooks Range, Alaska (ascending track, summer 2026, ~12-day
interferogram pairs), the computed rate map had a median around -2,500 to -3,000
mm/yr, with a range spanning roughly -17,000 to +9,200 mm/yr. Working backward,
that implies a typical (median) vertical displacement of ~10 cm within a single
12-day pair, applied across most of the scene.

Published InSAR literature on North Slope Alaska permafrost active-layer thaw
subsidence (Liu et al. 2010, JGR; Sentinel-1 studies, The Cryosphere 2024) reports:
- Seasonal (whole-summer, ~3-4 month) subsidence of 20-60 mm typically, up to
  ~80 mm in fast-degrading spots.
- Rates on the order of 2-18 mm/month.

Even the most extreme published rate (~18 mm/month) predicts only ~7 mm of LOS
displacement over a 12-day pair — roughly 2-3 orders of magnitude smaller than
what the current pipeline computes. This mismatch is not attributed to real ground
motion.

## Likely cause

1. **No reference-point phase calibration.** Unwrapped InSAR phase is only
   meaningful up to an arbitrary additive offset per interferogram (set by the
   unwrapping algorithm's own reference choice). Standard InSAR time-series
   workflows (SBAS and similar) remove this by subtracting a stable reference
   pixel/region's phase from every pair before any rate estimation.
   `process_displacement.py` does not currently do this — it feeds raw,
   unreferenced phase directly into the per-pixel stacking regression, so
   pair-to-pair offset differences (observed as several radians, i.e. multiple
   2π cycles, in per-layer median phase) dominate the resulting "rate."
2. **Preliminary/provisional calibration maturity.** The data are from
   `NISAR_L2_GUNW_PROVISIONAL_V1` — an early, not-yet-validated product from a
   recently launched mission. Orbital, tropospheric, and ionospheric correction
   terms are expected to carry larger residual errors than a validated product.
   The GUNW tropospheric phase screens are now subtracted by default
   (`nisar_corrections.py`, see the pipeline description). The ionospheric screen
   can be applied but is off by default, because it proved unreliable over the
   Greenland ice sheet (see below). Stacks built before this change are
   uncorrected; their stack file lacks the `ionosphere_corrected` /
   `troposphere_corrected` attributes.

## Why the angular distortion map is less affected

Angular distortion is the *spatial gradient* of the rate map. How much a given
artifact contaminates a gradient depends on its spatial wavelength relative to the
pixel spacing (80 m for NISAR GUNW):

- A single whole-scene constant phase offset cancels **exactly** under
  differencing, at any pixel separation — it contributes zero gradient.
- A smooth, long-wavelength trend (orbital/baseline/tropospheric residual, the
  type that dominates in this data — see below) looks nearly linear over a
  few-hundred-meter neighborhood, so it mostly adds a smooth background bias
  rather than sharp spatial contrast in the distortion map.
- A discrete jump at a `connectedComponents` (unwrapping-patch) boundary does
  **not** cancel — it lands as a sharp, spurious spike exactly where it occurs.

We checked this directly: for a representative granule/track over the current
~39 x 50 km canvas, `connectedComponents` is almost entirely one label (99.3% of
pixels in a single patch), yet the phase *within that single consistent patch*
still has a standard deviation of ~6.3 radians. This indicates the dominant
artifact here is a smooth, large-wavelength trend, not sharp unwrapping-boundary
jumps — consistent with what would be expected from residual
orbital/tropospheric/ionospheric calibration error rather than discrete unwrapping
resets, and consistent with the current distortion maps tracking real terrain
features rather than showing spurious sharp artifacts.

## Caveats for future work

- **Rate map absolute values are not usable quantitatively** until reference-point
  phase calibration is added and/or validated (non-provisional) NISAR data become
  available. Re-test this pipeline once validated data are released.
- **Angular distortion's absolute magnitude may still carry a background bias**
  from smooth large-scale trends (a ramp doesn't fully cancel, it just doesn't
  spike) — the *relative* spatial pattern is more trustworthy than the absolute
  value.
- **Steep terrain is a known exception.** Residual DEM/baseline-height-correlated
  phase error can be locally sharp exactly where real topography is sharp (steep
  valley walls, ridgelines) — precisely where a differential-settlement product
  might otherwise be most useful. Not yet specifically checked.
- **`connectedComponents` boundaries are not currently masked** — only a coherence
  threshold (`COHERENCE_MIN` in `process_displacement.py`) is applied. No sharp
  jump artifacts have been observed in current runs, but this should be monitored
  as the AOI, track, or terrain complexity changes (more fragmented
  `connectedComponents` would increase the risk).
- **Ionospheric scintillation** is a known small-scale L-band artifact source at
  auroral latitudes (northern Alaska sits under/near the auroral oval) and has not
  been independently verified as absent here. The GUNW split-spectrum ionosphere
  screen (available, but off by default) is spatially smoothed, so even when applied
  it would not remove small-scale scintillation.

## Ionospheric correction over the Greenland ice sheet (GUNW phase screen)

The GUNW product distributes an `ionospherePhaseScreen` (split-spectrum
estimate) that is not applied to `unwrappedPhase`. `nisar_corrections.py` can
subtract it. It was tested in `plot_dh_greenland.py` on the Greenland
comparison: the ATL15 1 km cell at the ICESat-2 crossover east of Point 660,
referenced to ice-free bedrock ~14 km to the west, over 25 twelve-day pairs on
tracks 088A, 160A and 141D (June–October 2026). The tropospheric screens were
applied at the same time.

### Observations

- **Effect on the five pairs valid on the ice.** All five fall in late
  September–October; the melt-season pairs are masked by coherence.
  Values are 1 km cell means in the pipeline's "apparent vertical" units
  (LOS / cos θ):

  | Pair | Uncorrected | Ionosphere + troposphere | Ionosphere, target − reference | Troposphere, target − reference |
  |---|---|---|---|---|
  | 088A 12–24 Sep | −36 cm | −65 cm | −29 cm | 0.0 cm |
  | 088A 24 Sep – 6 Oct | −35 cm | −67 cm | −32 cm | −0.5 cm |
  | 088A 6–18 Oct | −33 cm | −67 cm | −39 cm | +0.9 cm |
  | 141D 27 Sep – 9 Oct | +60 cm | +75 cm | +16 cm | −0.6 cm |
  | 160A 29 Sep – 11 Oct | −82 cm | −104 cm | −22 cm | −0.4 cm |

- **Size of the ionosphere correction.** Over all 25 pairs, the ionosphere
  correction between the ice target and the bedrock reference has a median
  magnitude of 39 cm (range −129 to +59 cm). The tropospheric correction is
  small: median 0.6 cm, at most 1.6 cm.
- **On stable bedrock.** Subtracting the ionosphere screen reduces the phase
  scatter compared with adding it in 18 of 25 pairs (median scatter
  0.72 rad raw, 0.53 rad subtracted, 1.10 rad added), which supports the sign
  convention. But subtracting it still *increases* the scatter compared with
  no correction in 9 of 25 pairs.
- **On the ice.** In every valid pair the ionosphere correction has the same
  sign as that track's flow-dominated signal: negative on both ascending
  tracks, positive on the descending track. It therefore widens the
  ascending/descending disagreement, and leaves the 088A/160A difference
  (~0.37 m) unchanged.

### Interpretation

- **Implausibly large.** A differential ionospheric delay of tens of
  centimetres, and more than a metre in one pair, over ~14 km is much larger
  than is plausible under most conditions. Corrections that consistently
  track the viewing geometry are also unlikely to be truly ionospheric,
  because the ionosphere is independent of ice flow.
- **Likely cause.** The split-spectrum estimate is probably unreliable over
  the ice:
  - the sub-band interferograms have even lower coherence than the full-band
    one (0.03–0.15 on the ice in the melt season, ~0.2–0.4 in the valid
    pairs), which promotes sub-band unwrapping errors;
  - the product fills gaps from the nearest valid value and smooths with an
    iterative Gaussian filter (`ionosphereFilling = nearest`,
    `ionosphereFiltering = Iterative gaussian filter` in the GUNW metadata),
    which spreads such errors across the ice margin.

  Motion-related errors leaking into the dispersive estimate on fast-moving
  ice would explain the sign pattern. That is a hypothesis; it was not tested
  here.
- **Not a sign error.** The sign convention is supported both by the ISCE3
  definition and by the bedrock test.

### Conclusions

- **Ionosphere off, troposphere on.** The ionosphere correction is **off by
  default** (`APPLY_IONOSPHERE_CORRECTION = False` in `download_nisar.py`
  and `plot_dh_greenland.py`). The tropospheric correction stays on: it is
  small and behaves consistently.
- **Before enabling it again,** test at least one of:
  - masking the screen with `ionospherePhaseScreenUncertainty` and/or
    coherence;
  - estimating the ionosphere only from coherent bedrock and interpolating a
    smooth surface across the ice;
  - checking the screen on the higher-coherence Alaska permafrost sites;
  - re-testing with validated (non-provisional) GUNW data.
- **What remains uncorrected.** The ionosphere is still a likely error source
  of centimetres to decimetres per pair, including at the Alaska sites. NISAR
  results on the ice remain qualitative.
- **Existing Alaska outputs.** The Alaska stacks and GeoTIFFs produced before
  this change were not re-processed. They carry no phase-screen corrections
  (their stack files lack the `ionosphere_corrected` /
  `troposphere_corrected` attributes).

## Current status

For the present proof-of-concept scope (small geographic extent, demonstrating the
processing pipeline end-to-end), the angular distortion output is considered
satisfactory. The rate map should be re-validated once calibrated NISAR data
become available.
