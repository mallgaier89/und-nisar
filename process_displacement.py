# process_displacement.py
# Loads the time-series stack saved by download_nisar.py, masks low-coherence
# pixels, converts unwrapped LOS phase to vertical displacement per pair,
# estimates a per-pixel mean displacement rate via InSAR stacking, derives an
# angular distortion field from that rate map, and writes both as GeoTIFFs.

import numpy as np                     # array operations
import h5py                            # read the intermediate stack file
import rasterio                        # GeoTIFF writer
from rasterio.transform import Affine  # geotransform from canvas origin/spacing
from rasterio.crs import CRS           # CRS from EPSG code
import matplotlib.pyplot as plt        # quick-look figure
from matplotlib.colors import TwoSlopeNorm  # zero-centered color scale for the diverging rate map


# =============================================================================
# Parameters
# Edit these values to change input/output paths or the masking/sign settings.
# =============================================================================
STACK_PATH                      = "./data/timeseries_stack.h5"                             # intermediate stack written by download_nisar.py
RATE_OUTPUT_PATH                = "./data/rate_vertical_mm_per_year.tif"                   # output: mean vertical displacement rate (unfiltered)
RATE_FILTERED_OUTPUT_PATH       = "./data/rate_vertical_mm_per_year_filtered.tif"           # output: mean vertical displacement rate (epoch-count filtered)
DISTORTION_OUTPUT_PATH          = "./data/angular_distortion_mm_per_yr_per_m.tif"           # output: angular distortion of the rate field (unfiltered)
DISTORTION_FILTERED_OUTPUT_PATH = "./data/angular_distortion_mm_per_yr_per_m_filtered.tif"  # output: angular distortion of the rate field (epoch-count filtered)
FIGURE_OUTPUT_PATH              = "./data/displacement_summary.png"                        # output: quick-look figure of both filtered maps
COHERENCE_MIN                   = 0.15   # pixels below this coherence magnitude are masked out before rate estimation
MIN_VALID_EPOCHS                = 2      # pixels with fewer valid epochs than this are masked out of the *_filtered outputs
SIGN_CONVENTION                 = -1.0  # +1.0 or -1.0 — flip if vertical rate sign disagrees with a known reference deformation
FIT_INTERCEPT                   = False  # True: per-pixel intercept+slope fit — each pixel absorbs its own average
                                         #   per-pair phase offset instead of it contaminating the rate. Use for
                                         #   current uncalibrated/provisional data (see LIMITATIONS.md).
                                         # False: original Sandwell & Price stacking, slope forced through the
                                         #   origin (no intercept) — only valid once each pair's phase is properly
                                         #   reference-calibrated so a zero-baseline pair truly has zero phase. Use
                                         #   once future validated/calibrated NISAR data are available.


# =============================================================================
# Load the intermediate stack
# Everything needed to go from phase to a georeferenced vertical rate map was
# captured per-granule by download_nisar.py and saved into one HDF5 file.
# =============================================================================
with h5py.File(STACK_PATH, "r") as hf:
    phase_3d       = hf["phase_3d"][:]                    # LOS unwrapped phase, radians (n, rows, cols)
    coherence_3d   = hf["coherence_3d"][:]                # coherence magnitude, 0-1 (n, rows, cols)
    incidence_3d   = hf["incidence_3d"][:]                # incidence angle, degrees (n, rows, cols)
    dates          = hf["dates"][:].astype(str)           # secondary acquisition dates (YYYYMMDD), one per layer
    baselines_days = hf["baselines_days"][:]              # temporal baseline per layer (days)
    canvas_x       = hf["canvas_x"][:]                    # canvas easting pixel-center coordinates (meters)
    canvas_y       = hf["canvas_y"][:]                    # canvas northing pixel-center coordinates (meters)
    dx             = float(hf.attrs["dx"])                # easting pixel spacing (meters)
    dy             = float(hf.attrs["dy"])                # northing pixel spacing (meters, signed)
    epsg           = int(hf.attrs["epsg"])                # projected CRS EPSG code
    wavelength_m   = float(hf.attrs["wavelength_m"])       # radar wavelength (meters)

print(f"Loaded stack: {phase_3d.shape} (granules x rows x cols)")
print(f"Dates: {list(dates)}  |  baselines (days): {list(baselines_days)}")


# =============================================================================
# Coherence mask
# Pixels below the coherence threshold are set to NaN per layer, so an
# unreliable pixel in one pair doesn't have to discard the whole column.
# =============================================================================
phase_masked = np.where(coherence_3d >= COHERENCE_MIN, phase_3d, np.nan)  # radians, NaN where masked

n_kept  = np.count_nonzero(coherence_3d >= COHERENCE_MIN)  # pixel-epochs surviving the mask
n_total = coherence_3d.size                                 # total pixel-epochs in the stack
print(f"Coherence mask (>= {COHERENCE_MIN}): kept {n_kept}/{n_total} pixel-epochs "
      f"({100 * n_kept / n_total:.1f}%)")


# =============================================================================
# Phase -> line-of-sight displacement -> vertical displacement
# LOS displacement follows the standard unwrapped-phase-to-range-change
# relation; projecting to vertical assumes purely vertical ground motion
# (the standard simplification for permafrost thaw-settlement studies).
# SIGN_CONVENTION is a single easily-flipped constant — verify against a
# known reference deformation event before trusting the absolute sign.
# =============================================================================
displacement_los      = SIGN_CONVENTION * (wavelength_m / (4 * np.pi)) * phase_masked  # meters
displacement_vertical = displacement_los / np.cos(np.radians(incidence_3d))            # meters, LOS -> vertical


# =============================================================================
# Per-pixel rate via InSAR stacking (Sandwell & Price, 1998), or per-pixel
# intercept+slope OLS if FIT_INTERCEPT is set — see the parameter comment
# above. Each pair's own temporal baseline is the independent variable, and
# each pair is treated independently: a simple chain of pairs rather than an
# overlapping network that would need a full SBAS-style inversion.
# =============================================================================
valid        = ~np.isnan(displacement_vertical)               # per-pixel, per-layer validity mask
baseline_col = baselines_days[:, None, None]                  # broadcast baseline over rows/cols

if FIT_INTERCEPT:
    # Unweighted OLS for displacement = intercept + rate * baseline. With an
    # intercept free to absorb each pixel's average phase offset, there's no
    # remaining physical argument for baseline-proportional weighting.
    t = np.where(valid, baseline_col, 0.0)          # baseline, zeroed where masked
    d = np.where(valid, displacement_vertical, 0.0)  # displacement, zeroed where masked
    n = valid.sum(axis=0).astype(float)              # per-pixel valid epoch count
    sum_t, sum_d   = t.sum(axis=0), d.sum(axis=0)
    sum_tt, sum_td = (t * t).sum(axis=0), (t * d).sum(axis=0)
    denom = n * sum_tt - sum_t ** 2  # 0 when n<2 or all valid baselines are identical
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_m_per_day = np.where(denom > 0, (n * sum_td - sum_t * sum_d) / denom, np.nan)  # OLS slope (meters/day)
        intercept_m    = np.where(denom > 0, (sum_d - rate_m_per_day * sum_t) / n, np.nan)  # OLS intercept (meters)
    print(f"Intercept+slope fit: intercept mean {np.nanmean(intercept_m) * 1000:.2f} mm "
          f"(min {np.nanmin(intercept_m) * 1000:.1f}, max {np.nanmax(intercept_m) * 1000:.1f})")
else:
    num = np.nansum(baseline_col * displacement_vertical, axis=0)  # sum(baseline * displacement), NaN-safe
    den = np.sum(np.where(valid, baseline_col ** 2, 0.0), axis=0)  # sum(baseline^2) over valid layers only
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_m_per_day = np.where(den > 0, num / den, np.nan)      # origin-forced weighted-LS slope (meters/day)

rate_mm_per_year = rate_m_per_day * 365.25 * 1000.0             # convert to mm/year

print(f"Rate map: {np.nanmin(rate_mm_per_year):.1f} to {np.nanmax(rate_mm_per_year):.1f} mm/yr "
      f"(mean {np.nanmean(rate_mm_per_year):.2f})")


# =============================================================================
# Minimum-epoch-count safeguard
# The fit above is solvable with very few valid observations (1 for the
# origin-forced fit, 2 for the intercept+slope fit) and has no way to flag a
# noisy or unrepresentative pixel on its own at that point count. Pixels with
# fewer than MIN_VALID_EPOCHS surviving the coherence mask are excluded from
# the *_filtered outputs; the unfiltered outputs above are kept as-is for
# full spatial coverage.
# =============================================================================
n_valid_epochs = valid.sum(axis=0)  # per-pixel count of coherence-surviving epochs

epoch_mask = n_valid_epochs >= MIN_VALID_EPOCHS
n_masked = np.count_nonzero(~epoch_mask)
n_pixels = n_valid_epochs.size
print(f"Epoch-count mask (>= {MIN_VALID_EPOCHS} valid epochs): dropped {n_masked}/{n_pixels} pixels "
      f"({100 * n_masked / n_pixels:.1f}%) from filtered outputs")

rate_mm_per_year_filtered = np.where(epoch_mask, rate_mm_per_year, np.nan)


# =============================================================================
# Angular distortion — spatial gradient of the rate map between neighboring
# pixels, a standard differential-settlement metric for infrastructure risk.
# Computed once from the unfiltered rate map and once from the filtered rate
# map; np.gradient naturally propagates NaN into any distortion pixel whose
# finite-difference computation touches a masked rate neighbor.
# =============================================================================
def compute_angular_distortion(rate_map):
    """Spatial gradient magnitude of a rate map, in mm/yr per meter."""
    drate_drow, drate_dcol = np.gradient(rate_map, abs(dy), abs(dx))  # mm/yr per meter, along each array axis
    if dy < 0:                       # canvas_y decreases with increasing row index
        drate_drow = -drate_drow     # flip so the gradient points toward increasing northing
    return np.hypot(drate_dcol, drate_drow)  # combined horizontal-plane gradient magnitude (mm/yr per m)


angular_distortion = compute_angular_distortion(rate_mm_per_year)
angular_distortion_filtered = compute_angular_distortion(rate_mm_per_year_filtered)

print(f"Angular distortion: max {np.nanmax(angular_distortion):.4f} mm/yr per m "
      f"(mean {np.nanmean(angular_distortion):.4f})")


# =============================================================================
# Write outputs as GeoTIFFs
# canvas_x / canvas_y hold pixel-center coordinates, so the raster origin
# (top-left corner of pixel [0, 0]) is offset by half a pixel in each axis.
# =============================================================================
transform = Affine(dx, 0.0, canvas_x[0] - dx / 2, 0.0, dy, canvas_y[0] - dy / 2)  # affine geotransform
crs = CRS.from_epsg(epsg)  # projected CRS matching the input stack


def write_geotiff(path, array):
    """Write a single-band float32 GeoTIFF using the shared transform/CRS above."""
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype=np.float32, crs=crs, transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(array.astype(np.float32), 1)
    print(f"Wrote {path}")


write_geotiff(RATE_OUTPUT_PATH, rate_mm_per_year)
write_geotiff(RATE_FILTERED_OUTPUT_PATH, rate_mm_per_year_filtered)
write_geotiff(DISTORTION_OUTPUT_PATH, angular_distortion)
write_geotiff(DISTORTION_FILTERED_OUTPUT_PATH, angular_distortion_filtered)


# =============================================================================
# Visualization
# Quick-look figure of both filtered maps side by side (epoch-count mask
# applied): a diverging blue/red scale centered on zero for the signed rate
# map, and a single-hue sequential scale for the (always non-negative)
# angular distortion magnitude. Unfiltered coverage remains available in the
# unfiltered GeoTIFFs written above.
# =============================================================================
extent = [                                       # corner-based extent for imshow (canvas_x/y hold pixel centers)
    canvas_x[0] - dx / 2, canvas_x[-1] + dx / 2,  # left, right
    canvas_y[-1] + dy / 2, canvas_y[0] - dy / 2,  # bottom, top
]

rate_limit = np.nanmax(np.abs(rate_mm_per_year_filtered))            # symmetric limits so zero sits at the midpoint
rate_norm  = TwoSlopeNorm(vmin=-rate_limit, vcenter=0.0, vmax=rate_limit)

distortion_limit = np.nanpercentile(angular_distortion_filtered, 98)  # clip a few outlier pixels for display only
print(f"Angular distortion color scale clipped at the 98th percentile "
      f"({distortion_limit:.4f} mm/yr per m) for display — underlying GeoTIFF is unclipped.")

fig, (ax_rate, ax_distortion) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

im_rate = ax_rate.imshow(rate_mm_per_year_filtered, extent=extent, origin="upper", cmap="RdBu_r", vmin=-2000, vmax=2000)
ax_rate.set_title("Mean vertical displacement rate")
ax_rate.set_xlabel("Easting (m)")
ax_rate.set_ylabel("Northing (m)")
fig.colorbar(im_rate, ax=ax_rate, label="mm/yr", shrink=0.85)


im_distortion = ax_distortion.imshow(1e-3*angular_distortion_filtered, extent=extent, origin="upper", cmap="Blues",
                                      vmin=0.0, vmax=1e-3*distortion_limit)
ax_distortion.set_title("Angular distortion")
ax_distortion.set_xlabel("Easting (m)")
ax_distortion.set_ylabel("Northing (m)")
fig.colorbar(im_distortion, ax=ax_distortion, label="1/yr", shrink=0.85)

fig.savefig(FIGURE_OUTPUT_PATH, dpi=150)
print(f"Wrote {FIGURE_OUTPUT_PATH}")
plt.show()
