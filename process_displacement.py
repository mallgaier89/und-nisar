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
STACK_PATH             = "./data/timeseries_stack.h5"                    # intermediate stack written by download_nisar.py
RATE_OUTPUT_PATH       = "./data/rate_vertical_mm_per_year.tif"          # output: mean vertical displacement rate
DISTORTION_OUTPUT_PATH = "./data/angular_distortion_mm_per_yr_per_m.tif"  # output: angular distortion of the rate field
FIGURE_OUTPUT_PATH     = "./data/displacement_summary.png"               # output: quick-look figure of both maps
COHERENCE_MIN          = 0.15   # pixels below this coherence magnitude are masked out before rate estimation
SIGN_CONVENTION        = -1.0  # +1.0 or -1.0 — flip if vertical rate sign disagrees with a known reference deformation


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
# Per-pixel rate via InSAR stacking (Sandwell & Price, 1998)
# Weighted least-squares slope through the origin using each pair's own
# temporal baseline as the independent variable. Appropriate here because the
# granules form a simple chain of independent interferogram pairs rather than
# an overlapping network that would need a full SBAS-style inversion.
# =============================================================================
valid        = ~np.isnan(displacement_vertical)               # per-pixel, per-layer validity mask
baseline_col = baselines_days[:, None, None]                  # broadcast baseline over rows/cols
num = np.nansum(baseline_col * displacement_vertical, axis=0)  # sum(baseline * displacement), NaN-safe
den = np.sum(np.where(valid, baseline_col ** 2, 0.0), axis=0)  # sum(baseline^2) over valid layers only
with np.errstate(invalid="ignore", divide="ignore"):
    rate_m_per_day = np.where(den > 0, num / den, np.nan)      # weighted-LS slope (meters/day)
rate_mm_per_year = rate_m_per_day * 365.25 * 1000.0             # convert to mm/year

print(f"Rate map: {np.nanmin(rate_mm_per_year):.1f} to {np.nanmax(rate_mm_per_year):.1f} mm/yr "
      f"(mean {np.nanmean(rate_mm_per_year):.2f})")


# =============================================================================
# Angular distortion — spatial gradient of the rate map between neighboring
# pixels, a standard differential-settlement metric for infrastructure risk.
# =============================================================================
drate_drow, drate_dcol = np.gradient(rate_mm_per_year, abs(dy), abs(dx))  # mm/yr per meter, along each array axis
if dy < 0:                       # canvas_y decreases with increasing row index
    drate_drow = -drate_drow     # flip so the gradient points toward increasing northing
angular_distortion = np.hypot(drate_dcol, drate_drow)  # combined horizontal-plane gradient magnitude (mm/yr per m)

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
write_geotiff(DISTORTION_OUTPUT_PATH, angular_distortion)


# =============================================================================
# Visualization
# Quick-look figure of both maps side by side: a diverging blue/red scale
# centered on zero for the signed rate map, and a single-hue sequential scale
# for the (always non-negative) angular distortion magnitude.
# =============================================================================
extent = [                                       # corner-based extent for imshow (canvas_x/y hold pixel centers)
    canvas_x[0] - dx / 2, canvas_x[-1] + dx / 2,  # left, right
    canvas_y[-1] + dy / 2, canvas_y[0] - dy / 2,  # bottom, top
]

rate_limit = np.nanmax(np.abs(rate_mm_per_year))                     # symmetric limits so zero sits at the midpoint
rate_norm  = TwoSlopeNorm(vmin=-rate_limit, vcenter=0.0, vmax=rate_limit)

distortion_limit = np.nanpercentile(angular_distortion, 98)          # clip a few outlier pixels for display only
print(f"Angular distortion color scale clipped at the 98th percentile "
      f"({distortion_limit:.4f} mm/yr per m) for display — underlying GeoTIFF is unclipped.")

fig, (ax_rate, ax_distortion) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

im_rate = ax_rate.imshow(rate_mm_per_year, extent=extent, origin="upper", cmap="RdBu_r", vmin=-2000, vmax=2000)
ax_rate.set_title("Mean vertical displacement rate")
ax_rate.set_xlabel("Easting (m)")
ax_rate.set_ylabel("Northing (m)")
fig.colorbar(im_rate, ax=ax_rate, label="mm/yr", shrink=0.85)


im_distortion = ax_distortion.imshow(1e-3**angular_distortion, extent=extent, origin="upper", cmap="Blues",
                                      vmin=0.0, vmax=1e-3*distortion_limit)
ax_distortion.set_title("Angular distortion")
ax_distortion.set_xlabel("Easting (m)")
ax_distortion.set_ylabel("Northing (m)")
fig.colorbar(im_distortion, ax=ax_distortion, label="1/yr", shrink=0.85)

fig.savefig(FIGURE_OUTPUT_PATH, dpi=150)
print(f"Wrote {FIGURE_OUTPUT_PATH}")
plt.show()
