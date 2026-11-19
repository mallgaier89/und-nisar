# nisar_corrections.py
# Ionospheric and tropospheric phase corrections for NISAR GUNW unwrapped
# interferograms, shared by the InSAR pipeline (download_nisar.py) and the
# Greenland ICESat-2 comparison (plot_dh_greenland.py). Importing this module
# has no side effects.
#
# The GUNW product distributes these layers but does NOT apply them to
# unwrappedPhase (ASF NISAR user guide, GUNW page):
#   - ionospherePhaseScreen (grids/.../unwrappedInterferogram/HH/, same 80 m
#     grid as the phase, radians): split-spectrum estimate of the dispersive
#     (ionospheric) phase component;
#   - wet/hydrostaticTroposphericPhaseScreen (metadata/radarGrid/, coarse 3D
#     cubes over ellipsoidal height x y x x, radians): ISCE3 writes them as
#     -4*pi/wavelength * (delay_reference - delay_secondary), i.e. the
#     tropospheric phase contained in a reference * conj(secondary)
#     interferogram (isce3 python/packages/nisar/workflows/troposphere.py).
# Both are therefore SUBTRACTED: corrected = unwrappedPhase - iono - tropo.
# The sign was also checked on stable bedrock (subtraction reduces the
# phase scatter there; see plot_dh_greenland.py).
#
# Troposphere depends strongly on height, so the cube is interpolated in 3D at
# each pixel's terrain height from the Copernicus GLO-90 DEM (cached tiles).
# The DEM is geoid-referenced (EGM2008) while the cube uses ellipsoidal
# heights; the difference (tens of metres vs 500 m cube spacing) is ignored.

from functools import lru_cache
import numpy as np
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

import sampling_common as sc             # Copernicus DEM tile cache (dem_mosaic)


G = "/science/LSAR/GUNW/"
IONO_PATH        = G + "grids/frequencyA/unwrappedInterferogram/HH/ionospherePhaseScreen"
PROJ_PATH        = G + "grids/frequencyA/unwrappedInterferogram/projection"
TROPO_PATHS      = (G + "metadata/radarGrid/wetTroposphericPhaseScreen",
                    G + "metadata/radarGrid/hydrostaticTroposphericPhaseScreen")
RG_X_PATH        = G + "metadata/radarGrid/xCoordinates"
RG_Y_PATH        = G + "metadata/radarGrid/yCoordinates"
RG_HEIGHTS_PATH  = G + "metadata/radarGrid/heightAboveEllipsoid"
DEM_CACHE_DIR    = "./data/basemap"      # Copernicus GLO-90 tiles (shared with the map basemaps)


def ionosphere_screen(h, rows, cols):
    """Ionosphere phase screen (radians) over the same window as a phase crop."""
    return h[IONO_PATH][rows, cols].astype(np.float64)


def _index_window(coords, lo, hi, pad=2):
    """Index slice of a monotonic coordinate vector bracketing [lo, hi] with `pad` extra nodes."""
    idx = np.flatnonzero((coords >= lo) & (coords <= hi))
    if idx.size == 0:                                      # points fall between two nodes
        idx = np.array([int(np.argmin(np.abs(coords - (lo + hi) / 2)))])
    return slice(max(int(idx.min()) - pad, 0), min(int(idx.max()) + pad + 1, len(coords)))


def troposphere_screen(h, x, y, z):
    """Wet + hydrostatic tropospheric phase (radians) at grid points x, y (GUNW CRS) and height z (m)."""
    rx, ry, heights = h[RG_X_PATH][:], h[RG_Y_PATH][:], h[RG_HEIGHTS_PATH][:]
    cs = _index_window(rx, np.nanmin(x), np.nanmax(x))
    rs = _index_window(ry, np.nanmin(y), np.nanmax(y))
    sub_x, sub_y = rx[cs], ry[rs]
    cube = sum(h[p][:, rs, cs].astype(np.float64) for p in TROPO_PATHS)
    if sub_y[0] > sub_y[-1]:                               # interpolator needs ascending axes
        sub_y, cube = sub_y[::-1], cube[:, ::-1, :]
    interp = RegularGridInterpolator((heights, sub_y, sub_x), cube, bounds_error=False, fill_value=None)
    zc = np.clip(z, heights[0], heights[-1])               # clamp to the cube's height range
    return interp(np.stack([zc, y, x], axis=-1))


@lru_cache(maxsize=8)
def _dem_for(bbox_key):
    return sc.dem_mosaic(bbox_key, DEM_CACHE_DIR)


def terrain_height(x, y, crs):
    """Copernicus GLO-90 terrain height (m, EGM2008 geoid) at grid points x, y in `crs`, bilinear."""
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    bbox = (np.floor(np.nanmin(lon) * 10) / 10 - 0.1, np.floor(np.nanmin(lat) * 10) / 10 - 0.1,
            np.ceil(np.nanmax(lon) * 10) / 10 + 0.1, np.ceil(np.nanmax(lat) * 10) / 10 + 0.1)
    dem, tf = _dem_for(bbox)
    col = (lon - tf.c) / tf.a - 0.5                        # fractional pixel indices (pixel-centre convention)
    row = (lat - tf.f) / tf.e - 0.5
    return map_coordinates(dem.astype(np.float64), [row, col], order=1, mode="nearest")


def correct_phase(h, phase, rows, cols, x, y, apply_iono=True, apply_tropo=True):
    """
    Corrected unwrapped phase for a crop phase = unwrappedPhase[rows, cols], with x, y the crop's
    1D pixel-centre coordinates. Returns (corrected phase, stats) where stats holds the median
    ionosphere and troposphere corrections over the window (radians, NaN if not applied).
    """
    corrected = phase.astype(np.float64)
    stats = {"iono_median_rad": np.nan, "tropo_median_rad": np.nan}
    if apply_iono:
        iono = ionosphere_screen(h, rows, cols)
        corrected = corrected - iono
        stats["iono_median_rad"] = float(np.nanmedian(iono))
    if apply_tropo:
        xx, yy = np.meshgrid(x, y)
        crs = f"EPSG:{int(h[PROJ_PATH][()])}"
        tropo = troposphere_screen(h, xx, yy, terrain_height(xx, yy, crs))
        corrected = corrected - tropo
        stats["tropo_median_rad"] = float(np.nanmedian(tropo))
    return corrected, stats
