# plot_sampling_greenland.py
# Proposal figures contrasting NISAR and ICESat-2 sampling on the Greenland
# ice sheet east of Point 660 (end of the road from Kangerlussuaq):
#   1. a three-panel sampling map —
#      A. region: NISAR frame borders, all six ICESat-2 nominal beam tracks,
#         crossovers, and the 10 x 10 km study area;
#      B. the study area with the ATL15 1 km (3-month) and 2.5 km (1-month)
#         grid cells and the ICESat-2 beams crossing it;
#      C. a ~1 km zoom at the sample point: NISAR 80 m pixels, ATL15 cell
#         edges, and ICESat-2 ATL06 segments (20 m posting);
#   2. a sampling timeline at the sample point (an ICESat-2 crossover inside
#      the study area): NISAR tracks (actual + nominal 12-day), the crossing
#      ICESat-2 RGTs (actual ATL06 + nominal 91-day), and the ATL15 epochs.
# CMR metadata (NISAR footprints, ATL06/ATL15 granules) needs no login and is
# cached under ./data/cmr_cache/. The ATL15 grid registration and epoch times
# are read from the real ATL15 files when an Earthdata login is stored
# (~/.netrc); otherwise a nominal grid/epoch layout is used and flagged.

import glob
import json
import os
import re
import numpy as np
import rasterio                          # GIMP ice-mask tiles
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

import icesat2_tracks as is2             # ICESat-2 nominal RGT helpers
import sampling_common as sc             # shared CMR / basemap / drawing helpers
from sampling_common import C_NISAR, C_IS2_ASC, C_IS2_DESC, INK, INK_2, INK_3, HALO, KM as km


# =============================================================================
# Parameters
# =============================================================================
SITE_NAME          = "Point 660"
FILE_TAG           = "Point660"
ROAD_END_LON       = -50.0478      # Point 660, end of the road (OpenStreetMap node 9171158090)
ROAD_END_LAT       = 67.1520
BOX_SIZE_M         = 10_000.0      # study area: north-up square, west edge at the road end, centred N-S on it
MAP_MARGIN_M       = 20_000.0      # regional panel extends this far beyond the study area
INSET_HALF_WIDTH_M = 500.0         # zoom panel half-width around the sample point
WINDOW_START       = "2025-06-01"  # timeline window
WINDOW_END         = "2026-12-31"
MELT_SEASON        = ("06-01", "08-31")  # MM-DD span shaded on the timeline (each year)
NISAR_SHORT_NAME   = "NISAR_L2_GUNW_PROVISIONAL_V1"
NISAR_REPEAT_DAYS  = 12
NISAR_OPS_START    = "2025-11-01"  # no nominal NISAR dates before science operations (first GUNW references: Nov 2025)
NISAR_GRID_CRS     = "EPSG:3413"   # GUNW geocoding grid over Greenland (as at Toolik); pixel edges on multiples of
NISAR_PIXEL_M      = 80.0          #   80 m (GUNW pixel centres sit at 40 m mod 80 m)
ATL15_CRS          = "EPSG:3413"
ATL15_GRIDS        = {             # product key -> (cell size m, epoch step, CMR title tag)
    "3mo_1km":   (1000.0, "3mo", "_3mo_1km_"),
    "1mo_2.5km": (2500.0, "1mo", "_1mo_2.5km_"),
}
ATL15_META_CACHE   = "./data/atl15_grid_meta.json"
GIMP_DIR           = "./data/gimp"    # cached GIMP land-ice mask tiles (NSIDC-0714 v1.2, Earthdata login)
GIMP_EPOCH         = "2020"           # GIMP mask epoch (2000, 2015 or 2020)
GIMP_RES           = "15m"            # GIMP tile resolution (15m or 30m tiles; 90m single mosaic)
ICE_MASK_RES_M     = 30.0             # resampling of the ice mask onto the map grid (m)
BASEMAP_DIR        = "./data/basemap"
BASEMAP_RES_M      = 100.0
BASEMAP_Z_FACTOR   = 2.0
CMR_CACHE_DIR      = "./data/cmr_cache"
REFRESH_CMR        = False
MAP_OUTPUT         = f"./data/{FILE_TAG} sampling_map"
TIMELINE_OUTPUT    = f"./data/{FILE_TAG} sampling_timeline"

C_ATL15 = INK_2  # ATL15 grid/epochs: neutral ink, distinct from the per-pass ICESat-2 colours

sc.apply_style()


# =============================================================================
# Study box and north-up map projection
# MAP_CRS is an azimuthal equidistant projection centred on the box centre,
# so the box is the square [-5, 5] km x [-5, 5] km and map axes are km
# east / north of the box centre. The box centre lies 5 km due east of the
# road end.
# =============================================================================
half_box = BOX_SIZE_M / 2
road_crs = f"+proj=aeqd +lat_0={ROAD_END_LAT} +lon_0={ROAD_END_LON} +datum=WGS84 +units=m +no_defs"
centre_lon, centre_lat = Transformer.from_crs(road_crs, "EPSG:4326", always_xy=True).transform(half_box, 0.0)
MAP_CRS = f"+proj=aeqd +lat_0={centre_lat} +lon_0={centre_lon} +datum=WGS84 +units=m +no_defs"

to_map = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
to_geo = Transformer.from_crs(MAP_CRS, "EPSG:4326", always_xy=True)
grid_to_map = Transformer.from_crs(ATL15_CRS, MAP_CRS, always_xy=True)  # ATL15 and NISAR share EPSG:3413
map_to_grid = Transformer.from_crs(MAP_CRS, ATL15_CRS, always_xy=True)

x_road, y_road = to_map.transform(ROAD_END_LON, ROAD_END_LAT)
left, right = -half_box - MAP_MARGIN_M, half_box + MAP_MARGIN_M
bottom, top = -half_box - MAP_MARGIN_M, half_box + MAP_MARGIN_M
geo_bbox = sc.geo_bbox_of(to_geo, left, bottom, right, top)
print(f"Box centre {centre_lat:.4f}N {centre_lon:.4f}E; road end at ({x_road:.0f}, {y_road:.0f}) m in map CRS")

hillshade, hillshade_extent = sc.load_hillshade(geo_bbox, MAP_CRS, left, right, bottom, top, BASEMAP_DIR,
                                                BASEMAP_RES_M, BASEMAP_Z_FACTOR)


# =============================================================================
# Ice-sheet margin: GIMP land-ice classification mask (Howat et al. 2014;
# NSIDC-0714 v1.2; 1 = land ice), epoch GIMP_EPOCH. The EPSG:3413 tiles
# covering the map are downloaded once (Earthdata login), mosaicked,
# cropped and resampled (nearest) onto a north-up MAP_CRS grid.
# =============================================================================
def load_ice_mask():
    os.makedirs(GIMP_DIR, exist_ok=True)
    pattern = os.path.join(GIMP_DIR, f"GimpIceMask_{GIMP_RES}_{GIMP_EPOCH}_tile*_v1.2.tif")
    if not glob.glob(pattern):
        import earthaccess
        earthaccess.login(strategy="netrc")
        granules = earthaccess.search_data(short_name="NSIDC-0714", bounding_box=geo_bbox,
                                           temporal=(f"{GIMP_EPOCH}-01-01", f"{GIMP_EPOCH}-12-31"))
        urls = [u for g in granules for u in g.data_links()
                if os.path.basename(u).startswith(f"GimpIceMask_{GIMP_RES}_{GIMP_EPOCH}_tile")]
        earthaccess.download(urls, GIMP_DIR)
    srcs = [rasterio.open(p) for p in sorted(glob.glob(pattern))]
    gx, gy = map_to_grid.transform(*sc.rect_outline(left, bottom, right, top))
    pad = 2 * ICE_MASK_RES_M
    try:
        mosaic, mosaic_tf = merge(srcs, bounds=(min(gx) - pad, min(gy) - pad, max(gx) + pad, max(gy) + pad))
    finally:
        for s in srcs:
            s.close()
    n_cols = int(np.ceil((right - left) / ICE_MASK_RES_M))
    n_rows = int(np.ceil((top - bottom) / ICE_MASK_RES_M))
    ice = np.zeros((n_rows, n_cols), np.uint8)
    reproject(mosaic[0].astype(np.uint8), ice, src_transform=mosaic_tf, src_crs=ATL15_CRS,
              dst_transform=rasterio.transform.from_origin(left, top, ICE_MASK_RES_M, ICE_MASK_RES_M),
              dst_crs=MAP_CRS, resampling=Resampling.nearest)
    xc = left + ICE_MASK_RES_M * (np.arange(n_cols) + 0.5)   # cell-centre coordinates for contouring
    yc = top - ICE_MASK_RES_M * (np.arange(n_rows) + 0.5)
    return (ice == 1), xc, yc


ice_mask, ice_x, ice_y = load_ice_mask()
box_cols = np.abs(ice_x) <= half_box
box_rows = np.abs(ice_y) <= half_box
print(f"GIMP {GIMP_EPOCH} ice mask: {100 * ice_mask.mean():.0f}% of map is ice; "
      f"{100 * ice_mask[np.ix_(box_rows, box_cols)].mean():.0f}% of the study area")


def draw_ice(ax, lw=1.4):
    """Light tint over the ice sheet and the margin line (GIMP mask boundary)."""
    ext = [(ice_x[0] - ICE_MASK_RES_M / 2) * km, (ice_x[-1] + ICE_MASK_RES_M / 2) * km,
           (ice_y[-1] - ICE_MASK_RES_M / 2) * km, (ice_y[0] + ICE_MASK_RES_M / 2) * km]
    ax.imshow(np.ma.masked_where(~ice_mask, np.ones_like(ice_mask, float)), cmap=ListedColormap(["white"]),
              alpha=0.35, extent=ext, origin="upper", interpolation="nearest", zorder=0.5)
    ax.contour(ice_x * km, ice_y * km, ice_mask.astype(float), levels=[0.5], colors=INK, linewidths=lw,
               zorder=3.5).set(path_effects=sc.LINE_HALO)


# =============================================================================
# CMR granule metadata (public), cached
# =============================================================================
bbox_param = ",".join(f"{v:.4f}" for v in geo_bbox)
temporal = f"{WINDOW_START}T00:00:00Z,2027-01-01T00:00:00Z"
nisar_entries = sc.cmr_search(CMR_CACHE_DIR, f"{FILE_TAG}_nisar_gunw.json", REFRESH_CMR,
                              short_name=NISAR_SHORT_NAME, bounding_box=bbox_param, temporal=temporal)
atl06_entries = sc.cmr_search(CMR_CACHE_DIR, f"{FILE_TAG}_atl06.json", REFRESH_CMR,
                              short_name="ATL06", bounding_box=bbox_param, temporal=temporal)
atl15_entries = sc.cmr_search(CMR_CACHE_DIR, f"{FILE_TAG}_atl15.json", REFRESH_CMR,
                              short_name="ATL15", bounding_box=bbox_param)
print(f"CMR: {len(nisar_entries)} NISAR GUNW, {len(atl06_entries)} ATL06, {len(atl15_entries)} ATL15 granules")


# =============================================================================
# ATL15 products: latest release per grid, its archive end, and the grid
# registration / epoch times. Real values come from the ATL15 file's
# delta_h/x, delta_h/y, delta_h/time (days since 2018-01-01), streamed with
# earthaccess when an Earthdata login is stored; otherwise nominal values:
# cell centres on whole multiples of the cell size and epochs at the start of
# each quarter / month.
# =============================================================================
RELEASE_RE = re.compile(r"ATL15_GL_(\d{4})_")


def read_atl15_meta(title):
    """x/y node spacing offset and epoch dates from the real ATL15 file, or None."""
    if os.path.exists(ATL15_META_CACHE):
        with open(ATL15_META_CACHE) as f:
            cache = json.load(f)
        if title in cache:
            return cache[title]
    netrc = [os.path.expanduser(p) for p in ("~/.netrc", "~/_netrc")]
    if not (any(os.path.exists(p) for p in netrc) or os.environ.get("EARTHDATA_USERNAME")):
        return None
    try:
        import earthaccess
        import h5py
        earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
        granule = earthaccess.search_data(short_name="ATL15", granule_name=title + "*", count=1)
        fh = earthaccess.open(granule)[0]  # streamed file object (not a context manager)
        try:
            with h5py.File(fh, "r") as h:
                x, y, t = h["delta_h/x"][:], h["delta_h/y"][:], h["delta_h/time"][:]
        finally:
            fh.close()
        epochs = [str(np.datetime64("2018-01-01") + np.timedelta64(int(round(d)), "D")) for d in t]
        meta = {"x0": float(x[0]), "y0": float(y[0]), "dx": float(x[1] - x[0]), "epochs": epochs}
    except Exception as err:  # network/auth/format problem: fall back to nominal
        print(f"  could not read ATL15 metadata for {title}: {err}")
        return None
    cache = {}
    if os.path.exists(ATL15_META_CACHE):
        with open(ATL15_META_CACHE) as f:
            cache = json.load(f)
    cache[title] = meta
    with open(ATL15_META_CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return meta


atl15 = {}
for key, (cell_m, step, tag) in ATL15_GRIDS.items():
    cands = [e for e in atl15_entries if tag in e["title"]]
    latest = max(cands, key=lambda e: RELEASE_RE.match(e["title"]).group(1))
    title = latest["title"].split(".nc")[0]
    meta = read_atl15_meta(title)
    archive_end = np.datetime64(latest["time_end"][:10])
    if meta:
        node_offset = meta["x0"] % cell_m              # cell centres sit at node_offset (mod cell size)
        epochs = np.array(meta["epochs"], dtype="datetime64[D]")
    else:
        node_offset = 0.0
        months = 3 if step == "3mo" else 1
        epochs = np.arange(np.datetime64("2019-01"), np.datetime64("2027-01"), months).astype("datetime64[D]")
    atl15[key] = {"cell": cell_m, "step": step, "title": title, "archive_end": archive_end,
                  "node_offset": node_offset, "epochs": epochs, "from_file": bool(meta)}
    print(f"ATL15 {key}: {title}, archive to {archive_end}, "
          f"{'grid/epochs from file' if meta else 'NOMINAL grid/epochs (no Earthdata login)'}")
atl15_nominal = not all(g["from_file"] for g in atl15.values())


def grid_lines(cell, offset, x_lo, x_hi, y_lo, y_hi, n=20):
    """
    Cell-edge lines of a square EPSG:3413 grid (cell centres at offset mod cell) covering a
    map-CRS box, returned as a list of (x, y) polylines in MAP_CRS.
    """
    gx, gy = map_to_grid.transform(*sc.rect_outline(x_lo, y_lo, x_hi, y_hi))
    first = lambda v: np.floor((v - offset - cell / 2) / cell) * cell + offset + cell / 2
    xs = np.arange(first(min(gx)), max(gx) + cell, cell)
    ys = np.arange(first(min(gy)), max(gy) + cell, cell)
    lines = []
    for xe in xs:
        lines.append(grid_to_map.transform(np.full(n, xe), np.linspace(ys[0], ys[-1], n)))
    for ye in ys:
        lines.append(grid_to_map.transform(np.linspace(xs[0], xs[-1], n), np.full(n, ye)))
    return lines


# =============================================================================
# ICESat-2 geometry. The sample point is an ascending x descending crossover
# inside the study box: RGT centrelines first; if none falls inside, the
# beam-pair centrelines (pairs at 0, +/-3.3 km); failing that, the nearest
# centreline crossover in the map.
# =============================================================================
passes = is2.project_passes(is2.load_rgts(*geo_bbox), MAP_CRS)
crossovers = [c for c in is2.find_crossovers(passes) if left <= c["x"] <= right and bottom <= c["y"] <= top]
in_box = lambda c: abs(c["x"]) <= half_box and abs(c["y"]) <= half_box

xo, xo_kind = None, ""
box_xo = [c for c in crossovers if in_box(c)]
if box_xo:
    xo, xo_kind = min(box_xo, key=lambda c: np.hypot(c["x"], c["y"])), "RGT centrelines"
    xo["off_asc"] = xo["off_desc"] = 0.0
else:
    pair_xo = []
    for pa in (p for p in passes if p.ascending):
        for pd in (p for p in passes if not p.ascending):
            for oa, (ax_, ay_) in is2.beam_tracks(pa, include_weak=False).items():
                for od, (dx_, dy_) in is2.beam_tracks(pd, include_weak=False).items():
                    qa = is2.Pass(pa.rgt, pa.lon, pa.lat, pa.t, True, ax_, ay_)
                    qd = is2.Pass(pd.rgt, pd.lon, pd.lat, pd.t, False, dx_, dy_)
                    for c in is2.find_crossovers([qa, qd]):
                        if in_box(c):
                            pair_xo.append({**c, "off_asc": oa, "off_desc": od})
    if pair_xo:
        xo, xo_kind = min(pair_xo, key=lambda c: np.hypot(c["x"], c["y"])), "beam pairs"
    else:
        xo, xo_kind = min(crossovers, key=lambda c: np.hypot(c["x"], c["y"])), "RGT centrelines (outside box)"
        xo["off_asc"] = xo["off_desc"] = 0.0
x0, y0 = xo["x"], xo["y"]
lon0, lat0 = to_geo.transform(x0, y0)
pair_name = lambda off: "centre pair" if abs(off) < 1 else f"{'left' if off > 0 else 'right'} pair"
print(f"{len(passes)} ICESat-2 passes, {len(crossovers)} centreline crossovers in map, {len(box_xo)} in box")
print(f"Sample point: {xo_kind} crossover RGT {xo['rgt_asc']:04d} ({pair_name(xo['off_asc'])}) x "
      f"{xo['rgt_desc']:04d} ({pair_name(xo['off_desc'])}) at {lat0:.4f}N {lon0:.4f}E")


# =============================================================================
# NISAR frames and acquisitions at the sample point
# =============================================================================
win_start, win_end = np.datetime64(WINDOW_START), np.datetime64(WINDOW_END)
frames, point_dates, nisar_archive_end = sc.scan_nisar(nisar_entries, to_map, Bbox([[left, bottom], [right, top]]),
                                                       x0, y0)
point_tracks = sorted(point_dates, key=lambda k: (k[1], k[0]))
print(f"NISAR frames over map: {sorted(frames)}")
print(f"NISAR tracks over sample point: {point_tracks}")


# =============================================================================
# Timeline rows
# =============================================================================
ops_start = np.datetime64(NISAR_OPS_START)
nisar_series = []
for track, direction in point_tracks:
    actual, nominal = sc.nisar_dates(point_dates[(track, direction)], win_start, win_end, NISAR_REPEAT_DAYS)
    nominal = nominal[nominal >= ops_start]
    nisar_series.append({"label": f"NISAR track {track:03d} {'asc' if direction == 'A' else 'desc'}",
                         "actual": actual, "nominal": nominal, "colour": C_NISAR,
                         "marker": "^" if direction == "A" else "v"})
    print(f"  T{track:03d}{direction}: {len(actual)} actual, {len(nominal)} nominal-only dates in window")

atl06_starts = sc.atl06_starts_by_rgt(atl06_entries)
is2_series = []
for rgt, t, asc in ((xo["rgt_asc"], xo["t_asc"], True), (xo["rgt_desc"], xo["t_desc"], False)):
    actual, nominal = sc.is2_dates(rgt, t, atl06_starts, WINDOW_START, WINDOW_END)
    is2_series.append({"label": f"ICESat-2 RGT {rgt:04d} {'asc' if asc else 'desc'}",
                       "actual": actual, "nominal": nominal, "colour": C_IS2_ASC if asc else C_IS2_DESC,
                       "marker": "^" if asc else "v"})

atl15_rows = []
for key, g in atl15.items():
    if g["from_file"]:
        # every epoch in the file is in the release; extend at the file's mean step for nominal future epochs
        released = g["epochs"]
        step_d = np.mean(np.diff(released).astype(int))
        k = np.arange(1, int((win_end - released[-1]).astype(int) / step_d) + 1)
        future = released[-1] + np.round(k * step_d).astype(int).astype("timedelta64[D]")
    else:
        released, future = g["epochs"][g["epochs"] <= g["archive_end"]], g["epochs"][g["epochs"] > g["archive_end"]]
    in_win = lambda e: e[(e >= win_start) & (e <= win_end)]
    s = {"actual": in_win(released), "nominal": in_win(future), "colour": C_ATL15, "marker": "D"}
    atl15_rows.append({"label": f"ATL15 {g['cell'] / 1000:g} km, {'3-month' if g['step'] == '3mo' else 'monthly'} epochs",
                       "series": [s], "brackets": True, "bracket_labels": g["step"] == "3mo"})

rows = ([{"label": s["label"], "series": [s], "brackets": False} for s in nisar_series]
        + [{"label": "NISAR all tracks combined", "series": nisar_series, "brackets": False, "combined": True}]
        + [{"label": s["label"], "series": [s], "brackets": True} for s in is2_series]
        + [{"label": "ICESat-2 both RGTs combined", "series": is2_series, "brackets": True, "combined": True}]
        + atl15_rows)


# =============================================================================
# Figure 1 — region / study area / 1 km zoom
# =============================================================================
fig = plt.figure(figsize=(7.5, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.28)
axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

# A — region
sc.draw_hillshade(axa, hillshade, hillshade_extent)
draw_ice(axa, lw=1.2)
to_data = lambda fx_, fy_: (left + fx_ * (right - left), bottom + fy_ * (top - bottom))  # axes fraction -> map m
keep_clear = ([(0.0, 0.0), (x_road, y_road), to_data(0.07, 0.9), to_data(0.07, 0.8)]   # box, Point 660, N arrow,
              + [to_data(f, 0.05) for f in np.linspace(0.05, 0.6, 6)])                 # corner note
covering = sc.draw_frame_borders(axa, frames, left, right, bottom, top, avoid=keep_clear)
sc.draw_beams(axa, passes, centre_lw=0.9, outer_lw=0.35, include_weak=False)
axa.scatter([c["x"] * km for c in crossovers], [c["y"] * km for c in crossovers], s=9, facecolor="white",
            edgecolor=INK, lw=0.5, zorder=6)
axa.add_patch(Rectangle((-half_box * km, -half_box * km), BOX_SIZE_M * km, BOX_SIZE_M * km, fill=False, ec=INK,
                        lw=1.3, zorder=7, path_effects=HALO))
axa.scatter([x_road * km], [y_road * km], marker="o", s=22, color=INK, edgecolor="white", lw=0.7, zorder=8)
axa.annotate(SITE_NAME, (x_road * km, y_road * km), xytext=(-5, 0), textcoords="offset points", ha="right",
             va="center", fontsize=7, color=INK, path_effects=HALO, zorder=8)
if covering:
    axa.text(0.02, 0.02, "Frames covering the whole map:\n" + ", ".join(covering), transform=axa.transAxes,
             fontsize=5.5, color=C_NISAR, path_effects=HALO, zorder=7)
sc.north_arrow(axa, 0.07, 0.95)
axa.set_xlim(left * km, right * km)
axa.set_ylim(bottom * km, top * km)
axa.set_aspect("equal")
axa.xaxis.set_major_locator(plt.MaxNLocator(4))
axa.yaxis.set_major_locator(plt.MaxNLocator(4))
axa.set_xlabel("km east of study-area centre")
axa.set_ylabel("km north of study-area centre")
axa.set_title("A  Region: NISAR frames, ICESat-2 tracks", loc="left", fontsize=8)

# B — study area: ATL15 grids, all six beams (pair beams resolvable at this scale)
sc.draw_hillshade(axb, hillshade, hillshade_extent)
draw_ice(axb, lw=1.6)
for lx, ly in grid_lines(NISAR_PIXEL_M, NISAR_PIXEL_M / 2, -half_box, half_box, -half_box, half_box):
    axb.plot(np.asarray(lx) * km, np.asarray(ly) * km, color=C_NISAR, lw=0.15, alpha=0.6, zorder=1.2)  # NISAR 80 m pixels
for key, lw, ls in (("3mo_1km", 0.5, "-"), ("1mo_2.5km", 1.1, (0, (5, 2)))):
    g = atl15[key]
    for lx, ly in grid_lines(g["cell"], g["node_offset"], -half_box, half_box, -half_box, half_box):
        axb.plot(np.asarray(lx) * km, np.asarray(ly) * km, color=C_ATL15, lw=lw, ls=ls, zorder=2)
sc.draw_beams(axb, passes, centre_lw=0.9, outer_lw=0.9)
axb.scatter([x0 * km], [y0 * km], marker="s", s=60, facecolor="none", edgecolor=INK, lw=1.2, zorder=7,
            path_effects=HALO)
axb.annotate("sample point", (x0 * km, y0 * km), xytext=(-8 if x0 > 0 else 8, 0), textcoords="offset points",
             ha="right" if x0 > 0 else "left", va="center", fontsize=6.5, color=INK, path_effects=HALO, zorder=7)
axb.set_xlim(-half_box * km, half_box * km)
axb.set_ylim(-half_box * km, half_box * km)
axb.set_aspect("equal")
axb.xaxis.set_major_locator(plt.MultipleLocator(2.5))
axb.yaxis.set_major_locator(plt.MultipleLocator(2.5))
axb.set_xlabel("km east")
for s in axb.spines.values():
    s.set_edgecolor(INK)
    s.set_linewidth(1.3)
axb.set_title(f"B  Study area {BOX_SIZE_M / 1000:g} × {BOX_SIZE_M / 1000:g} km: ATL15 and NISAR grids", loc="left", fontsize=8)
for ya in (half_box, -half_box):  # connectors: study box in A -> B
    fig.add_artist(ConnectionPatch(xyA=(half_box * km, ya * km), coordsA=axa.transData,
                                   xyB=(-half_box * km, ya * km), coordsB=axb.transData, color=INK_3, lw=0.6))

# C — 1 km zoom: NISAR 80 m pixels, ATL15 edges, crossing beam pairs with ATL06 points
zx0, zx1 = x0 - INSET_HALF_WIDTH_M, x0 + INSET_HALF_WIDTH_M
zy0, zy1 = y0 - INSET_HALF_WIDTH_M, y0 + INSET_HALF_WIDTH_M
axc.set_facecolor("#eef4fc")
for lx, ly in grid_lines(NISAR_PIXEL_M, NISAR_PIXEL_M / 2, zx0, zx1, zy0, zy1):  # pixel edges on multiples of 80 m
    axc.plot(lx, ly, color=C_NISAR, lw=0.5, alpha=0.7, zorder=1)
for key, lw, ls in (("3mo_1km", 1.0, "-"), ("1mo_2.5km", 1.6, (0, (5, 2)))):
    g = atl15[key]
    for lx, ly in grid_lines(g["cell"], g["node_offset"], zx0, zx1, zy0, zy1):
        axc.plot(lx, ly, color=C_ATL15, lw=lw, ls=ls, zorder=1.5)
half = is2.BEAM_SEPARATION_M / 2
for p in passes:
    if p.ascending and p.rgt == xo["rgt_asc"]:
        sc.draw_atl06_points(axc, p, (xo["off_asc"] - half, xo["off_asc"] + half), (zx0, zx1, zy0, zy1))
    elif not p.ascending and p.rgt == xo["rgt_desc"]:
        sc.draw_atl06_points(axc, p, (xo["off_desc"] - half, xo["off_desc"] + half), (zx0, zx1, zy0, zy1))
axc.set_xlim(zx0, zx1)
axc.set_ylim(zy0, zy1)
axc.set_aspect("equal")
axc.set_xticks([])
axc.set_yticks([])
for s in axc.spines.values():
    s.set_edgecolor(INK)
    s.set_linewidth(1.2)
axc.set_title("C  1 km zoom at sample point", loc="left", fontsize=8)
axc.plot([zx0 + 60, zx0 + 260], [zy0 + 60, zy0 + 60], color=INK, lw=2, solid_capstyle="butt", path_effects=HALO,
         zorder=5)
axc.text(zx0 + 160, zy0 + 75, "200 m", ha="center", va="bottom", fontsize=6.5, color=INK, path_effects=HALO, zorder=5)
for ya in (zy1, zy0):  # connectors: sample point in B -> C
    fig.add_artist(ConnectionPatch(xyA=(x0 * km, y0 * km), coordsA=axb.transData,
                                   xyB=(zx0, ya), coordsB=axc.transData, color=INK_3, lw=0.6))

handles = [Line2D([], [], color=C_IS2_ASC, lw=1.4, label="ICESat-2 asc beams (thick: centre pair on RGT)"),
           Line2D([], [], color=C_IS2_DESC, lw=1.4, label="ICESat-2 desc beams"),
           Line2D([], [], ls="", marker="o", mfc=C_IS2_ASC, mec="white", ms=4, label="ATL06 segment (20 m posting)"),
           Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=4, label="RGT crossover"),
           Patch(facecolor="none", edgecolor=C_NISAR, lw=0.9, label="NISAR frame border (dashed: descending)"),
           Patch(facecolor="#eef4fc", edgecolor=C_NISAR, lw=0.6, label=f"NISAR GUNW {NISAR_PIXEL_M:.0f} m pixel"),
           Line2D([], [], color=C_ATL15, lw=0.8, label="ATL15 1 km cell (3-month epochs)"),
           Line2D([], [], color=C_ATL15, lw=1.4, ls=(0, (5, 2)), label="ATL15 2.5 km cell (monthly epochs)"),
           Patch(facecolor="none", edgecolor=INK, lw=1.3, label=f"Study area ({BOX_SIZE_M / 1000:g} × {BOX_SIZE_M / 1000:g} km)"),
           Line2D([], [], color=INK, lw=1.4, path_effects=sc.LINE_HALO,
                  label=f"Ice-sheet margin (GIMP {GIMP_EPOCH}; ice lightened)")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
grid_note = ("ATL15 cell registration NOMINAL (cell centres on whole multiples of the cell size in EPSG:3413; "
             "not verified against the ATL15 file). " if atl15_nominal
             else f"ATL15 cells and epochs from {', '.join(g['title'] for g in atl15.values())}. ")
grid_note += (f"Ice margin: GIMP land-ice mask, {GIMP_EPOCH} epoch (Howat et al. 2014; NSIDC-0714 v1.2).\n")
fig.text(0.5, -0.115, f"Sample point: ICESat-2 {xo_kind} crossover, RGT {xo['rgt_asc']:04d} × {xo['rgt_desc']:04d}. "
         "North-up azimuthal equidistant projection; NISAR and ATL15 grids (EPSG:3413) appear rotated ~5°.\n"
         + grid_note + sc.DEM_ATTRIBUTION, ha="center", va="top", fontsize=5.8, color=INK_2, wrap=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.2)
sc.save_figure(fig, MAP_OUTPUT)


# =============================================================================
# Figure 2 — sampling timeline at the sample point
# Filled markers: acquired / in the archive (NISAR GUNW, ATL06 granules,
# ATL15 epochs within the latest release). Open markers: nominal schedule.
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
yr = np.arange(len(rows))[::-1]
for year in range(int(WINDOW_START[:4]), int(WINDOW_END[:4]) + 1):
    s0, s1 = np.datetime64(f"{year}-{MELT_SEASON[0]}"), np.datetime64(f"{year}-{MELT_SEASON[1]}")
    ax.axvspan(s0, s1, color="#efeee9", zorder=0, lw=0)
    ax.text(s0 + (s1 - s0) / 2, yr[0] + 0.75, "melt season", ha="center", va="bottom", fontsize=7, color=INK_2)
n_n, n_i = len(nisar_series), len(is2_series)
if ops_start > win_start:  # hatch the NISAR rows only
    y_lo, y_hi = yr[n_n] - 0.5, yr[0] + 0.5
    ax.add_patch(Rectangle((mdates.date2num(win_start), y_lo), mdates.date2num(ops_start) - mdates.date2num(win_start),
                           y_hi - y_lo, facecolor="none", edgecolor="#d9d8d3", hatch="////", lw=0, zorder=0.5))
    ax.text(win_start + (ops_start - win_start) / 2, (y_lo + y_hi) / 2, "before NISAR\nscience operations",
            ha="center", va="center", fontsize=6.5, color=INK_2, path_effects=HALO, zorder=4)
ax.axhline(yr[n_n] - 0.5, color=INK_3, lw=0.4)                 # NISAR tracks | NISAR combined
ax.axhline(yr[n_n + 1] + 0.5, color=INK_3, lw=0.8)             # NISAR | ICESat-2
ax.axhline(yr[n_n + 1 + n_i] - 0.5, color=INK_3, lw=0.4)       # ICESat-2 RGTs | combined
ax.axhline(yr[n_n + 2 + n_i] + 0.5, color=INK_3, lw=0.8)       # ICESat-2 tracks | ATL15
sc.draw_timeline_rows(ax, rows, yr)
ax.set_xlim(win_start, win_end + np.timedelta64(1, "D"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.grid(axis="x", which="both", color="#e4e3df", lw=0.6, zorder=0)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title(f"Sampling dates east of {SITE_NAME}: ICESat-2 crossover RGT {xo['rgt_asc']:04d} × "
             f"{xo['rgt_desc']:04d} ({lat0:.3f}°N, {abs(lon0):.3f}°W)", loc="left")
legend = [Line2D([], [], ls="", marker="^", color=INK_2, ms=5, label="acquired / in archive"),
          Line2D([], [], ls="", marker="^", mfc="white", mec=INK_2, ms=5, label="nominal schedule"),
          Line2D([], [], ls="", marker="^", color=INK_3, ms=5, label="ascending"),
          Line2D([], [], ls="", marker="v", color=INK_3, ms=5, label="descending"),
          Line2D([], [], ls="", marker="D", color=C_ATL15, ms=4.5,
                 label=f"ATL15 epoch{' (nominal timing)' if atl15_nominal else ''}")]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=7)
fig.subplots_adjust(left=0.24, right=0.88, top=0.9, bottom=0.2)
sc.save_figure(fig, TIMELINE_OUTPUT)
plt.show()
