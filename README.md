# NISAR InSAR Time-Series Displacement Pipeline

This repository is a proof-of-concept pipeline for turning NASA-ISRO SAR
(NISAR) interferometric products into per-pixel vertical displacement rate
and angular distortion maps, aimed at permafrost active-layer thaw/settlement
monitoring in northern Alaska. It has three stages, run in order:

```
search_nisar.py  →  download_nisar.py  →  process_displacement.py
```

`inspect_h5.py` is a standalone utility for browsing the internal structure
of a downloaded NISAR HDF5 file (useful when adapting the dataset paths in
`download_nisar.py` to a new product version).

## Pipeline

### 1. `search_nisar.py` — granule discovery

Queries NASA Earthdata's Common Metadata Repository (CMR) for NISAR granules
intersecting a bounding box and date range, without downloading anything.

- `search_nisar_granules(...)` authenticates with Earthdata via
  `earthaccess.login()` (environment variables, `~/.netrc`, or an interactive
  prompt, in that order), builds a CMR query from `short_name`,
  `bounding_box`, and `temporal` filters, and optionally applies an
  `orbit_direction` ('ascending'/'descending') filter. NISAR encodes pass
  direction as the CMR additional attribute `ASCENDING_DESCENDING`, which
  isn't one of `earthaccess`'s built-in keyword filters, so it's added as a
  raw CMR attribute query parameter.
- Returns a list of `earthaccess` `DataGranule` objects and prints a summary
  table (filename, size) so results can be sanity-checked before download.
- The `if __name__ == "__main__"` block at the bottom holds a set of
  commented-out site presets (test bounding boxes/date windows for several
  Brooks Range, Alaska locations) — uncomment one to run a search directly
  from this file.

### 2. `download_nisar.py` — download, crop, and stack

Runs the search above, then downloads and assembles the results into a
single time-series stack:

1. **Download** each matching granule one at a time via `earthaccess.download()`.
2. **Crop** each granule's unwrapped phase, coherence, and incidence angle to
   the search bounding box. The bbox is reprojected into the granule's native
   projected CRS (densifying its perimeter first, since a lat/lon rectangle
   is curved in projected space) rather than converting the swath to lon/lat,
   which matters near the poles where polar-stereographic swaths can be
   significantly rotated relative to lines of constant latitude/longitude.
3. **Frame edge mask** (`frame_edge_keep_mask`): NISAR GUNW granules are
   geocoded onto a rectangular grid, but the imaged swath is a rotated
   parallelogram within it. Adjacent frames along a track overlap, and the
   least reliable pixels are nearest the true swath boundary — not the
   rectangle's edge. This computes each valid pixel's distance to the
   nearest invalid/exterior pixel and nulls the outermost `EDGE_MASK_FRACTION`
   (default 10%) of the frame's valid area, following the swath's true
   shape regardless of how the search bbox happens to crop it.
4. **Incidence angle interpolation**: incidence angle is only tabulated on a
   coarse `(height, row, col)` radar grid, so it's bilinearly interpolated
   onto the full-resolution phase grid at the height nearest the scene's
   median reference terrain height.
5. **Temporal baseline**: each NISAR GUNW filename encodes a reference and
   secondary acquisition date; their difference (days) is stored per layer
   for use in the rate fit downstream.
6. **Canvas assembly**: builds a shared output grid from the union of all
   crops' actual projected extents (not a geographic reprojection), so every
   crop places onto the canvas at an exact integer pixel offset.
7. **Stack persistence** (`save_stack`): writes `phase_3d`, `coherence_3d`,
   `incidence_3d` (each `(n_granules, rows, cols)`), `dates`,
   `baselines_days`, canvas coordinates, and CRS/radar metadata to a single
   HDF5 file (`./data/timeseries_stack.h5`) for `process_displacement.py`.

Each layer in the resulting stack is an **independent, short (~12-day)
interferogram pair** — a sequential chain of pairwise granules, not a
cumulative single-master stack and not an overlapping SBAS-style network.

### 3. `process_displacement.py` — rate and distortion estimation

Loads the stack and produces two georeferenced GeoTIFF products, each in an
unfiltered and epoch-count-filtered version, plus a quick-look PNG:

1. **Coherence mask**: pixels below `COHERENCE_MIN` (default 0.15) are set to
   NaN per layer, so one unreliable pair doesn't discard an otherwise-good
   pixel's whole time series.
2. **Phase → displacement**: unwrapped LOS phase is converted to LOS range
   change (`SIGN_CONVENTION * wavelength / 4π * phase`) and projected to
   vertical assuming purely vertical motion — the standard simplification
   for permafrost thaw-settlement studies. `SIGN_CONVENTION` is a single
   flip-able constant; verify it against a known reference deformation event
   before trusting the absolute sign.
3. **Per-pixel rate fit** — two selectable modes via `FIT_INTERCEPT`:
   - **`FIT_INTERCEPT = False`** (default): the original Sandwell & Price
     (1998) stacking formula, `rate = Σ(baseline·displacement) / Σ(baseline²)`,
     forcing the fit through the origin. This is the textbook approach for a
     chain of independent short-baseline interferogram pairs, and is
     physically correct once each pair's phase is properly reference-point
     calibrated (a zero-baseline pair truly has zero phase).
   - **`FIT_INTERCEPT = True`**: unweighted per-pixel OLS,
     `displacement = intercept + rate · baseline`. Each pixel absorbs its own
     average per-pair phase offset in the intercept term instead of it
     contaminating the rate — switch to this mode for the current
     uncalibrated, provisional NISAR data (see [Limitations](#limitations)),
     and back to the origin-forced default once validated, calibrated NISAR
     data are available.
4. **Minimum-epoch-count safeguard**: the fit is mathematically solvable with
   very few points (1 for the origin-forced fit, 2 for the intercept fit),
   which is not enough to distinguish a real signal from a single bad
   unwrap or coherence-mask miss. Pixels with fewer than `MIN_VALID_EPOCHS`
   (default 2) surviving the coherence mask are excluded from the
   `*_filtered` outputs; unfiltered outputs retain full spatial coverage.
5. **Angular distortion**: the spatial gradient magnitude of the rate map
   between neighboring pixels — a standard differential-settlement metric
   for infrastructure risk — computed for both the unfiltered and filtered
   rate maps.
6. **Outputs** (all in `./data/`):
   - `rate_vertical_mm_per_year.tif` / `..._filtered.tif`
   - `angular_distortion_mm_per_yr_per_m.tif` / `..._filtered.tif`
   - `displacement_summary.png` — quick-look figure of the filtered rate and
     distortion maps side by side.

### Utility: `inspect_h5.py`

Opens a single downloaded NISAR HDF5 file and prints its full internal
group/dataset tree (paths, shapes, dtypes, attributes). Use it to confirm or
update the hardcoded HDF5 dataset paths in `download_nisar.py` when a new
NISAR product version changes its internal layout.

## Limitations

<!-- Content below is copied verbatim from LIMITATIONS.md; keep the two files in sync. -->

### Current limitations of the displacement/distortion products

This documents known limitations of the outputs produced by `download_nisar.py` +
`process_displacement.py` (`rate_vertical_mm_per_year.tif`,
`angular_distortion_mm_per_yr_per_m.tif`), as of the current NISAR data releases
(`NISAR_L2_GUNW_PROVISIONAL_V1`).

#### Summary

The vertical displacement **rate** map is not physically reliable in absolute terms
and should be treated as a methodology proof-of-concept rather than a quantitative
science product. The **angular distortion** map is less affected in practice and,
in the current test runs, tracks real geographic/terrain features coherently — it
is considered satisfactory for the present, small-scale (few-km) proof-of-concept
scope.

#### What we observed

For a test AOI in the Brooks Range, Alaska (ascending track, summer 2026, ~12-day
interferogram pairs), the computed rate map had a median around -2,500 to
-3,000 mm/yr, with a range spanning roughly -17,000 to +9,200 mm/yr. Working
backward, that implies a typical (median) vertical displacement of ~10 cm
within a single 12-day pair, applied across most of the scene.

Published InSAR literature on North Slope Alaska permafrost active-layer thaw
subsidence (Liu et al. 2010, JGR; Sentinel-1 studies, The Cryosphere 2024) reports:
- Seasonal (whole-summer, ~3-4 month) subsidence of 20-60 mm typically, up to
  ~80 mm in fast-degrading spots.
- Rates on the order of 2-18 mm/month.

Even the most extreme published rate (~18 mm/month) predicts only ~7 mm of LOS
displacement over a 12-day pair — roughly 2-3 orders of magnitude smaller than
what the current pipeline computes. This mismatch is not attributed to real ground
motion.

#### Likely cause

1. **No reference-point phase calibration.** Unwrapped InSAR phase is only
   meaningful up to an arbitrary additive offset per interferogram (set by the
   unwrapping algorithm's own reference choice). Standard InSAR time-series
   workflows (SBAS and similar) remove this by subtracting a stable reference
   pixel/region's phase from every pair before any rate estimation.
   `process_displacement.py` does not currently do this — it feeds raw,
   unreferenced phase directly into the per-pixel stacking regression, so
   pair-to-pair offset differences (observed as several radians, i.e. multiple
   2π cycles, in per-layer median phase) dominate the resulting "rate."
   (A per-pixel intercept term, selectable via `FIT_INTERCEPT`, mitigates this
   for the rate estimate but does not remove the underlying per-pair offset.)
2. **Preliminary/provisional calibration maturity.** The data are from
   `NISAR_L2_GUNW_PROVISIONAL_V1` — an early, not-yet-validated product from a
   recently launched mission. Orbital, tropospheric, and ionospheric correction
   terms are expected to carry larger residual errors than a validated product.

#### Why the angular distortion map is less affected

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

#### Caveats for future work

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
  been independently verified as absent here.

#### Current status

For the present proof-of-concept scope (small geographic extent, demonstrating the
processing pipeline end-to-end), the angular distortion output is considered
satisfactory. The rate map should be re-validated once calibrated NISAR data
become available.

## Citations

**Data product.** Outputs are derived from the NISAR L2 GUNW (Geocoded
Unwrapped Interferogram) product, distributed by the Alaska Satellite
Facility DAAC:

> NISAR Science Team / Alaska Satellite Facility DAAC. *NISAR L2 GUNW
> Provisional Product* (`NISAR_L2_GUNW_PROVISIONAL_V1`). NASA/JPL-Caltech and
> ISRO. Distributed by NASA Alaska Satellite Facility DAAC.
> https://www.earthdata.nasa.gov/data/catalog (search "NISAR GUNW").
>
> The prior Beta release of this product line (`NISAR_L2_GUNW_BETA_V1`) is
> registered under DOI [10.5067/NIL1GUNW-B1](https://doi.org/10.5067/NIL1GUNW-B1).
> The Provisional product's own DOI was not confirmed at the time of writing —
> check the dataset's [NASA Earthdata catalog page](https://www.earthdata.nasa.gov/data/catalog)
> for the current, authoritative citation before publishing results.

**Mission.** NISAR is a joint NASA/ISRO Earth-observing radar mission:

> Rosen, P. A., et al. *The NASA-ISRO Synthetic Aperture Radar (NISAR)
> Mission*. NASA Jet Propulsion Laboratory. See the mission overview at
> https://science.nasa.gov/mission/nisar/ and the NASA Technical Reports
> Server record at https://ntrs.nasa.gov/citations/20210012158.

**Processing workflow — InSAR stacking.** The origin-forced rate fit
(`FIT_INTERCEPT = False`) implements:

> Sandwell, D. T., and E. J. Price (1998), Phase gradient approach to
> stacking interferograms, *Journal of Geophysical Research: Solid Earth*,
> 103(B12), 30183–30204, doi:[10.1029/1998JB900008](https://doi.org/10.1029/1998JB900008).

**Permafrost subsidence context** (referenced in [Limitations](#limitations)):

> Liu, L., T. Zhang, and J. Wahr (2010), InSAR measurements of surface
> deformation over permafrost on the North Slope of Alaska, *Journal of
> Geophysical Research*, 115, F03023, doi:[10.1029/2009JF001547](https://doi.org/10.1029/2009JF001547).
>
> Sadeghi Chorsi, T., F. J. Meyer, and T. H. Dixon (2024), Toward long-term
> monitoring of regional permafrost thaw with satellite interferometric
> synthetic aperture radar, *The Cryosphere*, 18, 3723–3740,
> doi:[10.5194/tc-18-3723-2024](https://doi.org/10.5194/tc-18-3723-2024).
