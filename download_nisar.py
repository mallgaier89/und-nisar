# download_nisar.py
# Searches for NISAR granules via CMR, downloads each file one at a time,
# extracts the unwrapped phase cropped to the search bbox, collects all crops
# with their projected coordinates, then builds a shared canvas from the
# actual data extents and assembles a 3D time-series stack with a dates array.

import pathlib                                   # standard-library path handling
import re                                        # regex for parsing acquisition dates from filenames
from datetime import datetime                    # parsing acquisition dates to compute temporal baselines
import numpy as np                               # array operations and stacking
import h5py                                      # HDF5 file reader for NISAR data
import pyproj                                    # coordinate reference system transformations
from scipy.interpolate import RegularGridInterpolator  # coarse-grid → full-res incidence angle interpolation
from scipy.ndimage import distance_transform_edt       # distance-to-nearest-invalid-pixel, for edge masking
import earthaccess                               # NASA Earthdata download client
from search_nisar import search_nisar_granules   # CMR search function from search_nisar.py

C_LIGHT = 299_792_458.0  # speed of light, m/s — used to convert radar center frequency to wavelength


# =============================================================================
# Download function
# Wraps earthaccess.download() for a single granule.  Defined as a function
# because it is called once per loop iteration.
# =============================================================================
def download(granule, output_dir):
    """Download a single DataGranule to output_dir; return a list of Path objects."""
    local_files = earthaccess.download(          # fetch the granule's file(s) from Earthdata
        [granule],                               # one-element list — earthaccess expects a list
        local_path=str(output_dir),              # destination directory as a string
    )
    return [pathlib.Path(p) for p in local_files]  # normalise to Path objects before returning


# =============================================================================
# Frame edge masking
# NISAR GUNW granules are geocoded onto a rectangular grid, but the actual
# imaged swath is a rotated parallelogram within that rectangle — most of the
# grid outside the swath is NaN. Frames along a track overlap, and the least
# reliable pixels are the ones nearest the true swath boundary, so a fixed
# rectangular border trimmed from the grid's rectangle would barely touch the
# swath itself. Instead this measures each valid pixel's Euclidean distance
# (in pixels) to the nearest invalid/exterior pixel, and nulls the pixels
# closest to that boundary until their combined area is area_fraction of the
# frame's total valid area — following the swath's true shape, corners
# included, regardless of how a given search bbox happens to crop it.
# =============================================================================
def frame_edge_keep_mask(valid, area_fraction):
    """Boolean mask (True = keep) over a full-frame validity mask, nulling the area_fraction of valid pixels nearest the frame's true edge."""
    if area_fraction <= 0:
        return valid.copy()                                        # nothing to mask
    distance  = distance_transform_edt(valid)                       # px distance from each valid pixel to the nearest invalid pixel
    threshold = np.percentile(distance[valid], 100 * area_fraction)  # distance below which area_fraction of the valid area falls
    return valid & (distance > threshold)                            # False for the outermost area_fraction of the frame


# =============================================================================
# Stack persistence
# Writes the finished time-series stack to a single HDF5 file so downstream
# scripts (e.g. process_displacement.py) can pick up from here without
# re-downloading or re-cropping any granules.
# =============================================================================
def save_stack(path, phase_3d, coherence_3d, incidence_3d, dates, baselines_days,
                canvas_x, canvas_y, dx, dy, epsg, wavelength_m):
    """Persist the assembled time-series stack and its geolocation/radar metadata to an HDF5 file."""
    with h5py.File(path, "w") as hf:                        # create/overwrite the output file
        hf.create_dataset("phase_3d", data=phase_3d)             # LOS unwrapped phase, radians (n, rows, cols)
        hf.create_dataset("coherence_3d", data=coherence_3d)     # coherence magnitude, 0-1 (n, rows, cols)
        hf.create_dataset("incidence_3d", data=incidence_3d)     # incidence angle, degrees (n, rows, cols)
        hf.create_dataset("dates", data=np.array(dates, dtype="S8"))          # secondary acquisition dates (YYYYMMDD)
        hf.create_dataset("baselines_days", data=baselines_days)             # temporal baseline per layer (days)
        hf.create_dataset("canvas_x", data=canvas_x)                          # canvas easting coordinates (meters)
        hf.create_dataset("canvas_y", data=canvas_y)                          # canvas northing coordinates (meters)
        hf.attrs["dx"] = dx                                     # easting pixel spacing (meters)
        hf.attrs["dy"] = dy                                     # northing pixel spacing (meters)
        hf.attrs["epsg"] = epsg                                 # projected CRS EPSG code
        hf.attrs["wavelength_m"] = wavelength_m                 # radar wavelength (meters)
    print(f"Stack saved: {pathlib.Path(path).resolve()}")  # confirm the file landed on disk


# =============================================================================
# Parameters
# Edit these values to change the search region, product, or date window.
# =============================================================================
# # Toolik Lake
# SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
# LAT_MIN    = 68.6000            # southern boundary — Brooks Range, Alaska
# LAT_MAX    = 69.0000              # northern boundary — Brooks Range, Alaska
# LON_MIN    = -149.7047             # western boundary — Brooks Range, Alaska
# LON_MAX    = -149.0000            # eastern boundary — Brooks Range, Alaska
# DATE_START = "2026-06-01"           # start of search window
# DATE_END   = "2026-08-01"           # end of search window
# COUNT      = 100                     # cap at 50 results for a quick test run
# ORBIT_DIRECTION = 'ascending'              # 'ascending', 'descending', or None for both

# Site 05
SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
LAT_MIN    = 65.55         # southern boundary — Brooks Range, Alaska
LAT_MAX    = 66.0             # northern boundary — Brooks Range, Alaska
LON_MIN    = -150.0           # western boundary — Brooks Range, Alaska
LON_MAX    = -149.0          # eastern boundary — Brooks Range, Alaska
DATE_START = "2026-06-01"           # start of search window
DATE_END   = "2026-08-01"           # end of search window
COUNT      = 100                     # cap at 50 results for a quick test run
ORBIT_DIRECTION = 'ascending'              # 'ascending', 'descending', or None for both

EDGE_MASK_FRACTION = 0.10  # fraction of each frame's valid-data area to null out nearest its true swath edge (mitigates overlap discontinuities)

# # Site 15
# SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
# LAT_MIN    = 69.359601        # southern boundary — Brooks Range, Alaska
# LAT_MAX    = 69.794874            # northern boundary — Brooks Range, Alaska
# LON_MIN    = -149.138111          # western boundary — Brooks Range, Alaska
# LON_MAX    = -148.174568         # eastern boundary — Brooks Range, Alaska
# DATE_START = "2026-06-01"           # start of search window
# DATE_END   = "2026-08-01"           # end of search window
# COUNT      = 100                     # cap at 50 results for a quick test run
# ORBIT_DIRECTION = 'ascending' 

# # Yukon
# SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
# LAT_MIN    = 66.05        # southern boundary — Brooks Range, Alaska
# LAT_MAX    = 66.996237           # northern boundary — Brooks Range, Alaska
# LON_MIN    = -151.000000          # western boundary — Brooks Range, Alaska
# LON_MAX    = -149.847660         # eastern boundary — Brooks Range, Alaska
# DATE_START = "2026-06-01"           # start of search window
# DATE_END   = "2026-08-01"           # end of search window
# COUNT      = 100                     # cap at 50 results for a quick test run
# ORBIT_DIRECTION = 'ascending' 


LOCAL_PATH  = "./data"                  # local directory for downloaded files
STACK_OUTPUT_PATH = "./data/timeseries_stack.h5"  # intermediate stack file for process_displacement.py


# =============================================================================
# HDF5 dataset paths (NISAR GUNW Beta V1 internal structure)
# These paths locate the unwrapped phase and its pixel-center coordinate arrays
# inside the HDF5 file.  If a file raises a KeyError, open it in HDFView or
# with h5py to verify the actual paths in that product version.
# =============================================================================
PHASE_PATH   = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase"  # 2D phase array (radians)
X_COORD_PATH = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/xCoordinates"    # 1D easting array (meters, projected)
Y_COORD_PATH = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/yCoordinates"    # 1D northing array (meters, projected)
PROJ_PATH    = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/projection"          # EPSG code (uint32) for the projected CRS
COHERENCE_PATH          = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude"  # 2D coherence magnitude (0-1), same grid as phase
CENTER_FREQ_PATH        = "/science/LSAR/GUNW/grids/frequencyA/centerFrequency"                                # radar center frequency (Hz)
RADARGRID_INC_PATH      = "/science/LSAR/GUNW/metadata/radarGrid/incidenceAngle"                               # (height, row, col) incidence angle (degrees) on the coarse geometry grid
RADARGRID_X_PATH        = "/science/LSAR/GUNW/metadata/radarGrid/xCoordinates"                                 # 1D easting array for the coarse geometry grid (meters)
RADARGRID_Y_PATH        = "/science/LSAR/GUNW/metadata/radarGrid/yCoordinates"                                 # 1D northing array for the coarse geometry grid (meters, descending order)
RADARGRID_HEIGHTS_PATH  = "/science/LSAR/GUNW/metadata/radarGrid/heightAboveEllipsoid"                         # height levels at which incidence angle is tabulated (meters)
REF_TERRAIN_HEIGHT_PATH = "/science/LSAR/GUNW/metadata/processingInformation/parameters/reference/referenceTerrainHeight"  # terrain height profile along the reference orbit (meters)


# =============================================================================
# Output directory
# Create the download destination folder if it does not already exist.
# =============================================================================
output_dir = pathlib.Path(LOCAL_PATH)          # wrap the path string in a Path object
output_dir.mkdir(parents=True, exist_ok=True)  # create directory tree; no-op if already present
print(f"Download directory: {output_dir.resolve()}")  # log the resolved absolute path


# =============================================================================
# Search
# Call search_nisar_granules() to authenticate with Earthdata and query CMR.
# The returned list of DataGranule objects drives the download loop below.
# =============================================================================
granules = search_nisar_granules(  # query CMR for matching NISAR granules
    short_name = SHORT_NAME,       # NISAR collection identifier
    lat_min    = LAT_MIN,          # spatial filter — southern edge
    lat_max    = LAT_MAX,          # spatial filter — northern edge
    lon_min    = LON_MIN,          # spatial filter — western edge
    lon_max    = LON_MAX,          # spatial filter — eastern edge
    date_start = DATE_START,       # temporal filter — start date
    date_end   = DATE_END,         # temporal filter — end date
    count      = COUNT,            # cap on granules returned
    orbit_direction = ORBIT_DIRECTION,  # 'ascending', 'descending', or None
)


# =============================================================================
# Pre-loop setup
# crop_items collects each valid granule's phase crop together with the
# projected coordinates of its top-left pixel.  dx / dy are the pixel
# spacings and are the same for every NISAR GUNW granule.  tf_to_geo_stored
# keeps the projected→geographic transformer alive after the HDF5 file closes
# so it can be used in the post-loop canvas block.
# =============================================================================
crop_items       = []    # list of (phase_crop_2d, coherence_crop_2d, incidence_crop_2d, x_start, y_start) per valid granule
dates            = []    # secondary acquisition date (YYYYMMDD) per valid granule
baselines_days   = []    # temporal baseline (secondary date - reference date, days) per valid granule
dx               = None  # easting pixel spacing in meters — set from first valid granule
dy               = None  # northing pixel spacing in meters — set from first valid granule
tf_to_geo_stored = None  # projected → geographic transformer — set from first valid granule
epsg_stored      = None  # projected CRS EPSG code — set from first valid granule
wavelength_m     = None  # radar wavelength in meters — set from first valid granule
downloaded_paths = []    # local file paths for all downloaded files
total            = len(granules)  # total granules returned by the search


# =============================================================================
# Download loop
# Each iteration downloads one granule, opens it, and crops the unwrapped
# phase to the search bbox.  The crop and its projected starting coordinates
# are appended to crop_items for canvas assembly after the loop.
# =============================================================================
for i, granule in enumerate(granules):  # one granule per iteration

    print(f"\n--- Granule {i + 1} of {total} ---")  # progress marker for the console

    # --- pre-download filter (placeholder) -----------------------------------
    # Add metadata-based filtering here to skip granules before downloading.
    # Available without downloading: granule.size() → MB, granule['umm'] → full metadata dict.
    # Example:
    #   if granule.size() > 5000:
    #       print("  Skipping: exceeds size limit.")
    #       continue

    # --- download ------------------------------------------------------------
    print(f"  Downloading {granule.data_links()[0].split('/')[-1]} …")  # log filename before fetch
    local_files = download(granule, output_dir)  # download this granule; returns list of Path objects
    print(f"  Saved: {local_files[0].name}")     # confirm the file landed on disk

    # -------------------------------------------------------------------------
    # Post-download processing — open HDF5 and crop unwrapped phase to bbox
    # Coordinate arrays are read first (small 1D arrays) to locate bbox pixel
    # indices before touching the large phase raster.
    # -------------------------------------------------------------------------
    with h5py.File(local_files[0], "r") as hf:  # open HDF5 in read-only mode; closes automatically

        # --- read projection and build coordinate transformers ----------------
        proj_epsg  = int(hf[PROJ_PATH][()])                                           # EPSG code stored as uint32
        crs_proj   = pyproj.CRS.from_epsg(proj_epsg)                                 # projected CRS object
        crs_geo    = pyproj.CRS.from_epsg(4326)                                      # WGS84 geographic CRS
        tf_to_geo  = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)  # projected → degrees
        tf_to_proj = pyproj.Transformer.from_crs(crs_geo, crs_proj, always_xy=True)  # degrees → projected

        # --- read full 1D coordinate arrays (meters) -------------------------
        x_coords = hf[X_COORD_PATH][:]  # full 1D easting array for all columns (meters)
        y_coords = hf[Y_COORD_PATH][:]  # full 1D northing array for all rows (meters)

        # -------------------------------------------------------------------------
        # Reproject the search bbox into this granule's projected CRS, rather than
        # converting the grid to lon/lat at a single center row/column. Near the
        # poles a polar-stereographic swath can be significantly rotated relative
        # to lines of constant latitude/longitude, so a lon/lat test evaluated
        # only along the grid's center slice can miss a real overlap entirely.
        # The bbox perimeter is densified (a lat/lon rectangle maps to a curved
        # shape in projected space) and its bounding box is used for the crop
        # test below, run directly in the data's native, unrotated (x, y) space.
        # -------------------------------------------------------------------------
        n_perim   = 25  # points per edge — dense enough that the bounding box isn't underestimated
        perim_lon = np.concatenate([
            np.linspace(LON_MIN, LON_MAX, n_perim),  # bottom edge
            np.linspace(LON_MIN, LON_MAX, n_perim),  # top edge
            np.full(n_perim, LON_MIN),                # left edge
            np.full(n_perim, LON_MAX),                # right edge
        ])
        perim_lat = np.concatenate([
            np.full(n_perim, LAT_MIN),                 # bottom edge
            np.full(n_perim, LAT_MAX),                 # top edge
            np.linspace(LAT_MIN, LAT_MAX, n_perim),   # left edge
            np.linspace(LAT_MIN, LAT_MAX, n_perim),   # right edge
        ])
        perim_x, perim_y = tf_to_proj.transform(perim_lon, perim_lat)  # bbox perimeter in projected meters
        x_bbox_min, x_bbox_max = perim_x.min(), perim_x.max()          # projected-space bounding box
        y_bbox_min, y_bbox_max = perim_y.min(), perim_y.max()

        # --- find bounding-box pixel indices directly in projected space -----
        col_mask    = (x_coords >= x_bbox_min) & (x_coords <= x_bbox_max)  # True for columns inside bbox
        row_mask    = (y_coords >= y_bbox_min) & (y_coords <= y_bbox_max)  # True for rows inside bbox
        col_indices = np.where(col_mask)[0]                                 # integer column indices inside bbox
        row_indices = np.where(row_mask)[0]                                 # integer row indices inside bbox

        # --- skip tiles with no overlap --------------------------------------
        if col_indices.size == 0 or row_indices.size == 0:          # no pixels fall inside the bbox
            print("  No data within search box — skipping tile.")   # inform the user
            continue                                                 # move to the next granule

        col_start = int(col_indices[0])       # first column of the cropped region
        col_end   = int(col_indices[-1]) + 1  # one past the last column (exclusive)
        row_start = int(row_indices[0])       # first row of the cropped region
        row_end   = int(row_indices[-1]) + 1  # one past the last row (exclusive)

        # --- read the full-frame coherence array once ------------------------
        # Needed in full both for the coherence crop below and to locate this
        # frame's true valid-data footprint (NaN outside the imaged swath) for
        # edge masking — the swath boundary can lie anywhere relative to the
        # bbox crop, so it can't be determined from the crop alone.
        full_coherence = hf[COHERENCE_PATH][:]           # 2D float array (0-1), NaN outside the swath
        frame_valid    = ~np.isnan(full_coherence)        # True where this frame actually has data
        frame_keep     = frame_edge_keep_mask(frame_valid, EDGE_MASK_FRACTION)  # nulls the outer edge of the swath
        keep_crop      = frame_keep[row_start:row_end, col_start:col_end]       # same window as the crop below

        # --- read only the cropped phase slice from disk ---------------------
        # h5py slice indexing reads only the selected rectangle; the full
        # phase raster is never loaded into memory.
        phase_crop      = hf[PHASE_PATH][row_start:row_end, col_start:col_end]       # 2D float array (radians)
        phase_crop      = phase_crop.astype(np.float32)                             # consistent dtype across files
        coherence_crop  = full_coherence[row_start:row_end, col_start:col_end]       # 2D float array (0-1)
        coherence_crop  = coherence_crop.astype(np.float32)                         # consistent dtype across files

        # --- null out this frame's edge before it can reach the stack --------
        phase_crop[~keep_crop]     = np.nan
        coherence_crop[~keep_crop] = np.nan

        # -------------------------------------------------------------------------
        # Incidence angle crop — interpolated from the coarse radarGrid geometry
        # The incidence angle is only tabulated on a coarse (height, row, col) grid,
        # not on the full-resolution phase grid, so it must be interpolated onto the
        # exact pixel centers of the cropped phase region. Height varies the incidence
        # angle negligibly compared to across-swath position, so a single height
        # layer — the one nearest the median reference terrain height — is used.
        # -------------------------------------------------------------------------
        rg_heights   = hf[RADARGRID_HEIGHTS_PATH][:]                      # 1D height levels (meters)
        ref_height   = float(np.median(hf[REF_TERRAIN_HEIGHT_PATH][:]))   # representative terrain height for this scene
        height_idx   = int(np.argmin(np.abs(rg_heights - ref_height)))    # nearest tabulated height layer
        rg_x         = hf[RADARGRID_X_PATH][:]                            # coarse-grid easting axis (meters)
        rg_y         = hf[RADARGRID_Y_PATH][:]                            # coarse-grid northing axis (meters, descending)
        rg_incidence = hf[RADARGRID_INC_PATH][height_idx, :, :]           # (rows, cols) incidence angle (degrees) at that height

        incidence_interp = RegularGridInterpolator(                       # bilinear interpolator over the coarse grid
            (rg_y, rg_x), rg_incidence, bounds_error=False, fill_value=None
        )
        crop_y_mesh, crop_x_mesh = np.meshgrid(                           # full-res pixel centers within the crop
            y_coords[row_start:row_end], x_coords[col_start:col_end], indexing="ij"
        )
        incidence_crop = incidence_interp((crop_y_mesh, crop_x_mesh)).astype(np.float32)  # (rows, cols) degrees
        incidence_crop[~keep_crop] = np.nan  # apply the same frame-edge mask as phase/coherence

        # --- capture pixel spacing, transformer, CRS, and wavelength from first valid granule --
        # dx / dy, tf_to_geo, the EPSG code, and the radar wavelength are identical
        # for every NISAR GUNW granule (same grid, same CRS, same L-band radar), so
        # we only need to store them once.
        if dx is None:
            dx               = float(x_coords[1] - x_coords[0])  # easting step (meters, positive)
            dy               = float(y_coords[1] - y_coords[0])  # northing step (meters, sign = direction)
            tf_to_geo_stored = tf_to_geo                          # keep transformer alive after file closes
            epsg_stored      = proj_epsg                          # projected CRS EPSG code, for GeoTIFF export later
            center_freq_hz   = float(hf[CENTER_FREQ_PATH][()])    # radar center frequency (Hz)
            wavelength_m     = C_LIGHT / center_freq_hz            # radar wavelength (meters)

        # --- record crop with its projected top-left coordinate --------------
        x_start = float(x_coords[col_start])  # easting of the crop's first column
        y_start = float(y_coords[row_start])  # northing of the crop's first row
        crop_items.append((phase_crop, coherence_crop, incidence_crop, x_start, y_start))  # store for canvas assembly after loop
        n_valid_crop = int(np.count_nonzero(~np.isnan(coherence_crop)))  # valid pixels remaining in this crop after edge masking
        print(f"  Crop extracted: {phase_crop.shape}, "
              f"x={x_start:.0f}, y={y_start:.0f}, "
              f"{n_valid_crop} valid px after {EDGE_MASK_FRACTION:.0%} frame-edge mask")  # log crop size, origin, and edge mask

    # --- extract reference and secondary acquisition dates from filename -----
    # NISAR GUNW filenames contain four YYYYMMDDTHHMMSS datetime strings:
    # ref_start, ref_stop, secondary_start, secondary_stop.
    # The reference start date (index 0) and secondary start date (index 2)
    # identify this interferogram pair in time; their difference is the
    # temporal baseline used later for per-pixel rate estimation.
    date_matches    = re.findall(r"\d{8}T\d{6}", local_files[0].name)               # find all datetime strings
    reference_date  = date_matches[0][:8] if len(date_matches) >= 1 else "unknown"  # extract YYYYMMDD
    secondary_date  = date_matches[2][:8] if len(date_matches) >= 3 else "unknown"  # extract YYYYMMDD
    if reference_date != "unknown" and secondary_date != "unknown":
        baseline_days = (datetime.strptime(secondary_date, "%Y%m%d")
                          - datetime.strptime(reference_date, "%Y%m%d")).days       # temporal baseline in days
    else:
        baseline_days = None

    # --- accumulate results --------------------------------------------------
    dates.append(secondary_date)          # record the secondary acquisition date
    baselines_days.append(baseline_days)  # record the temporal baseline for this pair
    downloaded_paths.extend(local_files)  # record the local file path

    # --- post-processing filter (placeholder) --------------------------------
    # Add content-based filtering here (e.g., discard low-coherence granules)
    # before moving to the next iteration.


# =============================================================================
# Canvas assembly
# Build a shared canvas from the actual projected extents of all collected
# crops.  This avoids any geographic → projected conversion and guarantees
# every crop fits exactly within the canvas at a non-negative integer offset.
# The canvas spans the union of all crop extents; areas not covered by a given
# granule remain NaN in that granule's layer.
# =============================================================================
all_x_first = [item[3]                              for item in crop_items]  # easting of each crop's first col
all_x_last  = [item[3] + (item[0].shape[1]-1) * dx  for item in crop_items]  # easting of each crop's last col
all_y_first = [item[4]                              for item in crop_items]  # northing of each crop's first row
all_y_last  = [item[4] + (item[0].shape[0]-1) * dy  for item in crop_items]  # northing of each crop's last row

x_canvas_start = min(all_x_first)   # westernmost easting across all crops
x_canvas_end   = max(all_x_last)    # easternmost easting across all crops

if dy < 0:   # descending y — first row is northernmost (most positive)
    y_canvas_start = max(all_y_first)   # northernmost northing across all crops
    y_canvas_end   = min(all_y_last)    # southernmost northing across all crops
else:        # ascending y — first row is southernmost (least positive)
    y_canvas_start = min(all_y_first)   # southernmost northing across all crops
    y_canvas_end   = max(all_y_last)    # northernmost northing across all crops

canvas_cols = int(round((x_canvas_end   - x_canvas_start) / dx))        + 1  # number of columns
canvas_rows = int(round(abs(y_canvas_end - y_canvas_start) / abs(dy)))  + 1  # number of rows
canvas_x    = x_canvas_start + np.arange(canvas_cols) * dx    # 1D easting array for canvas columns
canvas_y    = y_canvas_start + np.arange(canvas_rows) * dy    # 1D northing array for canvas rows

print(f"\nCanvas: {canvas_rows} rows × {canvas_cols} cols, "
      f"pixel spacing {abs(dx):.0f} m × {abs(dy):.0f} m")    # log canvas dimensions

# compute 2D pixel-center lon/lat for the full canvas
x_mesh, y_mesh = np.meshgrid(canvas_x, canvas_y)              # 2D grids of projected coordinates
lons, lats     = tf_to_geo_stored.transform(x_mesh, y_mesh)   # 2D lon/lat arrays for every pixel center
lons           = lons.astype(np.float32)                        # store as float32
lats           = lats.astype(np.float32)                        # store as float32


# =============================================================================
# Time-series stack
# Pre-allocate the 3D array, then place each crop onto its canvas layer at
# the integer offset determined by its projected starting coordinate.
# Shape: (n_granules, canvas_rows, canvas_cols).  NaN marks pixels not covered
# by a given granule.  dates is a 1D numpy array aligned with axis 0.
# =============================================================================
phase_3d      = np.full((len(crop_items), canvas_rows, canvas_cols), np.nan, dtype=np.float32)  # pre-allocate with NaN
coherence_3d  = np.full((len(crop_items), canvas_rows, canvas_cols), np.nan, dtype=np.float32)  # pre-allocate with NaN
incidence_3d  = np.full((len(crop_items), canvas_rows, canvas_cols), np.nan, dtype=np.float32)  # pre-allocate with NaN
dates          = np.array(dates)                                                            # 1D YYYYMMDD date array
baselines_days = np.array(baselines_days, dtype=np.float64)                                  # 1D temporal baseline array (days)

for k, (phase_crop, coherence_crop, incidence_crop, x_start, y_start) in enumerate(crop_items):  # place each crop onto its canvas layer
    col_offset = int(round((x_start - canvas_x[0]) / dx))         # exact integer column offset within canvas
    row_offset = int(round((y_start - canvas_y[0]) / dy))         # exact integer row offset within canvas
    row_slice = slice(row_offset, row_offset + phase_crop.shape[0])  # row slice in canvas
    col_slice = slice(col_offset, col_offset + phase_crop.shape[1])  # column slice in canvas
    phase_3d[k, row_slice, col_slice]      = phase_crop       # time index for this granule
    coherence_3d[k, row_slice, col_slice]  = coherence_crop
    incidence_3d[k, row_slice, col_slice]  = incidence_crop
    print(f"  [{k+1}] placed at offset ({row_offset}, {col_offset}), "
          f"crop {phase_crop.shape}, date {dates[k]}")             # confirm placement

print(f"\nTime-series stack shape : {phase_3d.shape}  (granules × rows × cols)")             # report dimensions
print(f"Acquisition dates       : {dates}")                                                   # list all dates
print(f"Longitude range         : {lons.min():.6f} → {lons.max():.6f}  ({lons.shape[1]} cols)")  # spatial extent
print(f"Latitude  range         : {lats.min():.6f} → {lats.max():.6f}  ({lats.shape[0]} rows)")  # spatial extent
print(f"\nFinished. {len(downloaded_paths)} file(s) saved to {output_dir.resolve()}")         # final status


# =============================================================================
# Persist the stack
# Save the finished stack (and everything process_displacement.py needs to
# turn it into displacement products) to a single HDF5 file.
# =============================================================================
save_stack(
    STACK_OUTPUT_PATH,
    phase_3d=phase_3d,
    coherence_3d=coherence_3d,
    incidence_3d=incidence_3d,
    dates=dates,
    baselines_days=baselines_days,
    canvas_x=canvas_x,
    canvas_y=canvas_y,
    dx=dx,
    dy=dy,
    epsg=epsg_stored,
    wavelength_m=wavelength_m,
)
