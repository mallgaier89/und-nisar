# download_nisar.py
# Searches for NISAR granules via CMR, downloads each file one at a time,
# extracts the unwrapped phase cropped to the search bbox, collects all crops
# with their projected coordinates, then builds a shared canvas from the
# actual data extents and assembles a 3D time-series stack with a dates array.

import pathlib                                   # standard-library path handling
import re                                        # regex for parsing acquisition dates from filenames
import numpy as np                               # array operations and stacking
import h5py                                      # HDF5 file reader for NISAR data
import pyproj                                    # coordinate reference system transformations
import earthaccess                               # NASA Earthdata download client
from search_nisar import search_nisar_granules   # CMR search function from search_nisar.py


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
# Parameters
# Edit these values to change the search region, product, or date window.
# =============================================================================
SHORT_NAME  = "NISAR_L2_GUNW_BETA_V1"  # NISAR Level-2 Geocoded Unwrapped Interferogram (Beta V1)
LAT_MIN     = 67.412962                 # southern boundary — Brooks Range, Alaska
LAT_MAX     = 68.731326                 # northern boundary — Brooks Range, Alaska
LON_MIN     = -150.190912               # western boundary — Brooks Range, Alaska
LON_MAX     = -149.539300               # eastern boundary — Brooks Range, Alaska
DATE_START  = "2025-10-01"              # start of the temporal search window
DATE_END    = "2025-11-19"              # end of the temporal search window
COUNT       = 50                        # maximum granules to retrieve from CMR
LOCAL_PATH  = "./data"                  # local directory for downloaded files


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
)


# =============================================================================
# Pre-loop setup
# crop_items collects each valid granule's phase crop together with the
# projected coordinates of its top-left pixel.  dx / dy are the pixel
# spacings and are the same for every NISAR GUNW granule.  tf_to_geo_stored
# keeps the projected→geographic transformer alive after the HDF5 file closes
# so it can be used in the post-loop canvas block.
# =============================================================================
crop_items       = []    # list of (phase_crop_2d, x_start, y_start) per valid granule
dates            = []    # secondary acquisition date (YYYYMMDD) per valid granule
dx               = None  # easting pixel spacing in meters — set from first valid granule
dy               = None  # northing pixel spacing in meters — set from first valid granule
tf_to_geo_stored = None  # projected → geographic transformer — set from first valid granule
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

        # --- read projection and build coordinate transformer ----------------
        proj_epsg  = int(hf[PROJ_PATH][()])                                           # EPSG code stored as uint32
        crs_proj   = pyproj.CRS.from_epsg(proj_epsg)                                 # projected CRS object
        crs_geo    = pyproj.CRS.from_epsg(4326)                                      # WGS84 geographic CRS
        tf_to_geo  = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)  # projected → degrees

        # --- read full 1D coordinate arrays (meters) -------------------------
        x_coords = hf[X_COORD_PATH][:]  # full 1D easting array for all columns (meters)
        y_coords = hf[Y_COORD_PATH][:]  # full 1D northing array for all rows (meters)

        # --- convert 1D coordinate arrays to approximate degrees -------------
        # Each axis is converted at the centre value of the other axis.
        # Accurate to within one pixel for small AOIs; avoids a full 2D meshgrid.
        y_centre   = float(np.mean(y_coords))                                # fixed northing for x → lon
        x_centre   = float(np.mean(x_coords))                                # fixed easting for y → lat
        lons_1d, _ = tf_to_geo.transform(x_coords,                           # convert each easting …
                                          np.full_like(x_coords, y_centre))  # … at the centre northing
        _, lats_1d = tf_to_geo.transform(np.full_like(y_coords, x_centre),   # convert each northing …
                                          y_coords)                           # … at the centre easting

        # --- find bounding-box pixel indices in degree-space -----------------
        col_mask    = (lons_1d >= LON_MIN) & (lons_1d <= LON_MAX)  # True for columns inside bbox
        row_mask    = (lats_1d >= LAT_MIN) & (lats_1d <= LAT_MAX)  # True for rows inside bbox
        col_indices = np.where(col_mask)[0]                         # integer column indices inside bbox
        row_indices = np.where(row_mask)[0]                         # integer row indices inside bbox

        # --- skip tiles with no overlap --------------------------------------
        if col_indices.size == 0 or row_indices.size == 0:          # no pixels fall inside the bbox
            print("  No data within search box — skipping tile.")   # inform the user
            continue                                                 # move to the next granule

        col_start = int(col_indices[0])       # first column of the cropped region
        col_end   = int(col_indices[-1]) + 1  # one past the last column (exclusive)
        row_start = int(row_indices[0])       # first row of the cropped region
        row_end   = int(row_indices[-1]) + 1  # one past the last row (exclusive)

        # --- read only the cropped phase slice from disk ---------------------
        # h5py slice indexing reads only the selected rectangle; the full
        # raster is never loaded into memory.
        phase_crop = hf[PHASE_PATH][row_start:row_end, col_start:col_end]  # 2D float array (radians)
        phase_crop = phase_crop.astype(np.float32)                          # consistent dtype across files

        # --- capture pixel spacing and transformer from first valid granule --
        # dx / dy and tf_to_geo are identical for every NISAR GUNW granule
        # (same grid, same CRS), so we only need to store them once.
        if dx is None:
            dx               = float(x_coords[1] - x_coords[0])  # easting step (meters, positive)
            dy               = float(y_coords[1] - y_coords[0])  # northing step (meters, sign = direction)
            tf_to_geo_stored = tf_to_geo                          # keep transformer alive after file closes

        # --- record crop with its projected top-left coordinate --------------
        x_start = float(x_coords[col_start])  # easting of the crop's first column
        y_start = float(y_coords[row_start])  # northing of the crop's first row
        crop_items.append((phase_crop, x_start, y_start))  # store for canvas assembly after loop
        print(f"  Crop extracted: {phase_crop.shape}, "
              f"x={x_start:.0f}, y={y_start:.0f}")         # log crop size and projected origin

    # --- extract secondary acquisition date from filename --------------------
    # NISAR GUNW filenames contain four YYYYMMDDTHHMMSS datetime strings:
    # ref_start, ref_stop, secondary_start, secondary_stop.
    # The secondary start date (index 2) identifies this interferogram in time.
    date_matches   = re.findall(r"\d{8}T\d{6}", local_files[0].name)               # find all datetime strings
    secondary_date = date_matches[2][:8] if len(date_matches) >= 3 else "unknown"  # extract YYYYMMDD

    # --- accumulate results --------------------------------------------------
    dates.append(secondary_date)          # record the secondary acquisition date
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
all_x_first = [item[1]                              for item in crop_items]  # easting of each crop's first col
all_x_last  = [item[1] + (item[0].shape[1]-1) * dx  for item in crop_items]  # easting of each crop's last col
all_y_first = [item[2]                              for item in crop_items]  # northing of each crop's first row
all_y_last  = [item[2] + (item[0].shape[0]-1) * dy  for item in crop_items]  # northing of each crop's last row

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
phase_3d = np.full((len(crop_items), canvas_rows, canvas_cols), np.nan, dtype=np.float32)  # pre-allocate with NaN
dates    = np.array(dates)                                                                  # 1D YYYYMMDD date array

for k, (phase_crop, x_start, y_start) in enumerate(crop_items):  # place each crop onto its canvas layer
    col_offset = int(round((x_start - canvas_x[0]) / dx))         # exact integer column offset within canvas
    row_offset = int(round((y_start - canvas_y[0]) / dy))         # exact integer row offset within canvas
    phase_3d[k,                                                    # time index for this granule
             row_offset : row_offset + phase_crop.shape[0],        # row slice in canvas
             col_offset : col_offset + phase_crop.shape[1]] = phase_crop  # column slice in canvas
    print(f"  [{k+1}] placed at offset ({row_offset}, {col_offset}), "
          f"crop {phase_crop.shape}, date {dates[k]}")             # confirm placement

print(f"\nTime-series stack shape : {phase_3d.shape}  (granules × rows × cols)")             # report dimensions
print(f"Acquisition dates       : {dates}")                                                   # list all dates
print(f"Longitude range         : {lons.min():.6f} → {lons.max():.6f}  ({lons.shape[1]} cols)")  # spatial extent
print(f"Latitude  range         : {lats.min():.6f} → {lats.max():.6f}  ({lats.shape[0]} rows)")  # spatial extent
print(f"\nFinished. {len(downloaded_paths)} file(s) saved to {output_dir.resolve()}")         # final status
