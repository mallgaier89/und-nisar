# plot_sampling_toolik.py
# Proposal figures contrasting NISAR and ICESat-2 sampling at the Toolik site:
#   1. a sampling map — NISAR frame borders and all six ICESat-2 nominal beam
#      ground tracks with ascending x descending crossovers, plus a ~1 km zoom
#      panel at the sample point contrasting the NISAR 80 m GUNW pixel grid
#      with ICESat-2 ATL06 segments;
#   2. a sampling timeline — acquisition dates at the sample point (an
#      ICESat-2 crossover inside the displacement map) for every NISAR track
#      covering it and both crossing ICESat-2 RGTs, each individually and
#      combined (actual archive acquisitions + nominal repeat schedule).
# NISAR footprints and ATL06 granule lists come from NASA CMR (public
# metadata, no Earthdata login needed) and are cached as JSON under
# ./data/cmr_cache/ so reruns work offline. Shared helpers live in
# sampling_common.py.

import numpy as np
import rasterio                          # displacement-rate GeoTIFF: map extent and NISAR pixel grid
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, PathPatch
from matplotlib.path import Path

import icesat2_tracks as is2             # ICESat-2 nominal RGT helpers
import sampling_common as sc             # shared CMR / basemap / drawing helpers
from sampling_common import C_NISAR, C_IS2_ASC, C_IS2_DESC, INK, INK_2, INK_3, HALO, KM as km


# =============================================================================
# Parameters
# =============================================================================
SITE_NAME          = "Toolik"
RATE_PATH          = "./data/Toolik rate_vertical_mm_per_year.tif"  # defines map extent + NISAR 80 m pixel grid (EPSG:3413); values unused
SITE_LON           = -149.5869     # site centre (Toolik search bbox); the sample point is the ICESat-2
SITE_LAT           = 68.6521       #   crossover inside the displacement map nearest to it
WINDOW_START       = "2026-01-01"  # timeline window
WINDOW_END         = "2026-12-31"
THAW_SEASON        = ("06-01", "09-30")  # MM-DD span shaded on the timeline
NISAR_SHORT_NAME   = "NISAR_L2_GUNW_PROVISIONAL_V1"
NISAR_REPEAT_DAYS  = 12
MAP_MARGIN_M       = 20_000.0      # map extends this far beyond the displacement map on each side
# North-up map projection: azimuthal equidistant centred on the site, so map axes are km east / north
# of the site centre. (The NISAR GUNW grid, EPSG:3413 polar stereographic, has its y axis along the
# -45 deg meridian; at Toolik true north points ~104 deg clockwise from its "up", so plotting in
# EPSG:3413 would rotate the map.)
MAP_CRS            = f"+proj=aeqd +lat_0={SITE_LAT} +lon_0={SITE_LON} +datum=WGS84 +units=m +no_defs"
BASEMAP_DIR        = "./data/basemap"  # cached Copernicus GLO-90 DEM tiles for the hillshade basemap
BASEMAP_RES_M      = 100.0         # hillshade grid spacing on the map (m)
BASEMAP_Z_FACTOR   = 2.0           # vertical exaggeration of the hillshade (low-relief tundra)
INSET_HALF_WIDTH_M = 500.0         # zoom panel half-width around the sample point
NHD_WATERBODY_URL  = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12"  # USGS NHD Waterbody - Large Scale
LAKE_MIN_KM2       = 0.05          # smallest lake drawn (km^2); NHD FTYPE 390 = lake/pond, 436 = reservoir
LAKE_CACHE         = f"./data/nhd/{SITE_NAME}_waterbodies.geojson"
LAKE_LABEL         = "Toolik Lake"  # NHD GNIS name to label on the map
C_LAKE_FILL, C_LAKE_EDGE = "#bccfdc", "#5d7c90"  # muted gray-blue: water, kept distinct from NISAR blue
CMR_CACHE_DIR      = "./data/cmr_cache"
REFRESH_CMR        = False         # True: re-query CMR even if a cache file exists
MAP_OUTPUT         = f"./data/{SITE_NAME} sampling_map"       # .png and .pdf are written
TIMELINE_OUTPUT    = f"./data/{SITE_NAME} sampling_timeline"

sc.apply_style()


# =============================================================================
# Displacement-rate map (native NISAR grid) and map extent
# The rate map stays on its native EPSG:3413 grid (rl/rb/rr/rt, rate_tf);
# everything drawn on the map is in the north-up MAP_CRS. The map extent is
# the rate map's footprint in MAP_CRS plus MAP_MARGIN_M.
# =============================================================================
with rasterio.open(RATE_PATH) as src:
    rate = src.read(1, masked=True)
    rate_tf, rate_crs = src.transform, src.crs.to_string()
    rl, rb, rr, rt = src.bounds
px = rate_tf.a  # pixel size (m)

to_map = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
to_geo = Transformer.from_crs(MAP_CRS, "EPSG:4326", always_xy=True)
native_to_map = Transformer.from_crs(rate_crs, MAP_CRS, always_xy=True)
map_to_native = Transformer.from_crs(MAP_CRS, rate_crs, always_xy=True)

fx, fy = native_to_map.transform(*sc.rect_outline(rl, rb, rr, rt))
left, right = min(fx) - MAP_MARGIN_M, max(fx) + MAP_MARGIN_M
bottom, top = min(fy) - MAP_MARGIN_M, max(fy) + MAP_MARGIN_M
geo_bbox = sc.geo_bbox_of(to_geo, left, bottom, right, top)  # W, S, E, N
print(f"Rate map {rate.shape} px ({rate_crs}); map lon/lat bbox {np.round(geo_bbox, 3)}")

hillshade, hillshade_extent = sc.load_hillshade(geo_bbox, MAP_CRS, left, right, bottom, top, BASEMAP_DIR,
                                                BASEMAP_RES_M, BASEMAP_Z_FACTOR)

# Lakes: USGS NHD waterbody polygons (lakes/ponds, reservoirs) >= LAKE_MIN_KM2, public ArcGIS service, cached
lake_features = sc.arcgis_geojson(NHD_WATERBODY_URL, geo_bbox, f"AREASQKM >= {LAKE_MIN_KM2} AND FTYPE IN (390, 436)",
                                  LAKE_CACHE, out_fields="GNIS_NAME,FTYPE,AREASQKM")
lakes = [(f["properties"].get("GNIS_NAME"), sc.polygon_path(f["geometry"], to_map)) for f in lake_features]
print(f"NHD: {len(lakes)} lakes >= {LAKE_MIN_KM2} km^2 over the map")


def draw_lakes(ax, scale, box):
    """Lakes overlapping box = (x0, x1, y0, y1) in map metres, drawn in axes units of `scale` metres."""
    view = Bbox([[box[0], box[2]], [box[1], box[3]]])
    for _, path in lakes:
        if path.intersects_bbox(view):
            patch = PathPatch(Path(path.vertices * scale, path.codes), facecolor=C_LAKE_FILL, edgecolor=C_LAKE_EDGE,
                              lw=0.3, zorder=0.5)
            patch.set_in_layout(False)  # parts outside the axes must not widen the saved figure
            ax.add_patch(patch)


# =============================================================================
# CMR granule search (public metadata), cached to JSON
# =============================================================================
bbox_param = ",".join(f"{v:.4f}" for v in geo_bbox)
nisar_entries = sc.cmr_search(CMR_CACHE_DIR, f"{SITE_NAME}_nisar_gunw.json", REFRESH_CMR, short_name=NISAR_SHORT_NAME,
                              bounding_box=bbox_param, temporal="2025-10-01T00:00:00Z,2027-01-01T00:00:00Z")
atl06_entries = sc.cmr_search(CMR_CACHE_DIR, f"{SITE_NAME}_atl06.json", REFRESH_CMR, short_name="ATL06",
                              bounding_box=bbox_param, temporal="2025-10-01T00:00:00Z,2027-01-01T00:00:00Z")
print(f"CMR: {len(nisar_entries)} NISAR GUNW granules, {len(atl06_entries)} ATL06 granules over the map")


# =============================================================================
# ICESat-2 geometry: nominal RGT passes and crossovers. The sample point is
# the crossover inside the displacement map nearest the site centre.
# =============================================================================
passes = is2.project_passes(is2.load_rgts(*geo_bbox), MAP_CRS)
crossovers = [c for c in is2.find_crossovers(passes) if left <= c["x"] <= right and bottom <= c["y"] <= top]


def in_rate_map(c):
    xn, yn = map_to_native.transform(c["x"], c["y"])
    return rl <= xn <= rr and rb <= yn <= rt


xo = min((c for c in crossovers if in_rate_map(c)), key=lambda c: np.hypot(c["x"], c["y"]))  # site = map origin
x0, y0 = xo["x"], xo["y"]
lon0, lat0 = to_geo.transform(x0, y0)
print(f"{len(passes)} ICESat-2 passes, {len(crossovers)} crossovers in map")
print(f"Sample point: crossover RGT {xo['rgt_asc']:04d} x {xo['rgt_desc']:04d} at {lat0:.4f}N {lon0:.4f}E, "
      f"{np.hypot(x0, y0) / 1000:.1f} km from site centre")


# =============================================================================
# NISAR frames and acquisitions at the sample point
# =============================================================================
win_start, win_end = np.datetime64(WINDOW_START), np.datetime64(WINDOW_END)
frames, point_dates, archive_end = sc.scan_nisar(nisar_entries, to_map, Bbox([[left, bottom], [right, top]]), x0, y0)
point_tracks = sorted(point_dates, key=lambda k: (k[1], k[0]))
print(f"NISAR frames over map: {sorted(frames)}")
print(f"NISAR tracks over sample point: {point_tracks}")


# =============================================================================
# Dates at the sample point (timeline rows)
# =============================================================================
atl06_starts = sc.atl06_starts_by_rgt(atl06_entries)

nisar_series = []
for track, direction in point_tracks:
    actual, nominal = sc.nisar_dates(point_dates[(track, direction)], win_start, win_end, NISAR_REPEAT_DAYS)
    nisar_series.append({"label": f"NISAR track {track:03d} {'asc' if direction == 'A' else 'desc'}",
                         "actual": actual, "nominal": nominal, "colour": C_NISAR,
                         "marker": "^" if direction == "A" else "v"})
    print(f"  T{track:03d}{direction}: {len(actual)} actual, {len(nominal)} nominal-only dates in window")

is2_series = []
for rgt, t, asc in ((xo["rgt_asc"], xo["t_asc"], True), (xo["rgt_desc"], xo["t_desc"], False)):
    actual, nominal = sc.is2_dates(rgt, t, atl06_starts, WINDOW_START, WINDOW_END)
    is2_series.append({"label": f"ICESat-2 RGT {rgt:04d} {'asc' if asc else 'desc'}",
                       "actual": actual, "nominal": nominal, "colour": C_IS2_ASC if asc else C_IS2_DESC,
                       "marker": "^" if asc else "v"})

rows = ([{"label": s["label"], "series": [s], "brackets": False} for s in nisar_series]
        + [{"label": "NISAR all tracks combined", "series": nisar_series, "brackets": False, "combined": True}]
        + [{"label": s["label"], "series": [s], "brackets": True} for s in is2_series]
        + [{"label": "ICESat-2 both RGTs combined", "series": is2_series, "brackets": True, "combined": True}])


# =============================================================================
# Figure 1 — sampling map + zoom panel
# =============================================================================
fig = plt.figure(figsize=(7.5, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08)
ax = fig.add_subplot(gs[0])
axz = fig.add_subplot(gs[1])
sc.draw_hillshade(ax, hillshade, hillshade_extent)
draw_lakes(ax, km, (left, right, bottom, top))
for name, path in lakes:  # label the named site lake beside its outline
    if name == LAKE_LABEL:
        ax.add_patch(PathPatch(Path(path.vertices * km, path.codes), facecolor=C_LAKE_FILL, edgecolor=INK, lw=0.7,
                               path_effects=sc.LINE_HALO, zorder=6))  # the labelled lake stays visible above the tracks
        v = path.vertices
        ax.annotate(name, (v[:, 0].min() * km, v[:, 1].mean() * km), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5, fontstyle="italic", color=INK, path_effects=HALO, zorder=7)
covering = sc.draw_frame_borders(ax, frames, left, right, bottom, top, avoid=[(x0, y0)])
sc.draw_beams(ax, passes)

cx = np.array([c["x"] for c in crossovers]) * km
cy = np.array([c["y"] for c in crossovers]) * km
ax.scatter(cx, cy, s=12, facecolor="white", edgecolor=INK, lw=0.6, zorder=6)
ax.scatter([x0 * km], [y0 * km], marker="s", s=80, facecolor="none", edgecolor=INK, lw=1.3, zorder=7,
           path_effects=HALO)
ax.annotate("sample point", (x0 * km, y0 * km), xytext=(8, -10), textcoords="offset points", ha="left",
            color=INK, fontsize=7, path_effects=HALO, zorder=7)
if covering:
    ax.text(0.02, 0.02, "Frames covering the whole map: " + ", ".join(covering), transform=ax.transAxes,
            fontsize=6, color=C_NISAR, path_effects=HALO, zorder=7)
sc.north_arrow(ax)
ax.set_xlim(left * km, right * km)
ax.set_ylim(bottom * km, top * km)
ax.set_aspect("equal")
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_xlabel(f"km east of {SITE_NAME} site centre")
ax.set_ylabel(f"km north of {SITE_NAME} site centre")
ax.set_title(f"NISAR frames and ICESat-2 beam tracks over {SITE_NAME}", loc="left")

# zoom panel: outlines of the NISAR 80 m pixels (the rate map grid), centre-pair beams of both crossing RGTs,
# ATL06 segment centres with ~11 m footprints. Pixel corners are transformed from the
# native EPSG:3413 grid into MAP_CRS, so the cells appear rotated as they truly are.
zx0, zx1 = x0 - INSET_HALF_WIDTH_M, x0 + INSET_HALF_WIDTH_M
zy0, zy1 = y0 - INSET_HALF_WIDTH_M, y0 + INSET_HALF_WIDTH_M
xn0, yn0 = map_to_native.transform(x0, y0)
pad = 1.5 * INSET_HALF_WIDTH_M  # covers the rotated zoom box
r0, c0 = rasterio.transform.rowcol(rate_tf, xn0 - pad, yn0 + pad)
r1, c1 = rasterio.transform.rowcol(rate_tf, xn0 + pad, yn0 - pad)
r0, c0 = max(r0, 0), max(c0, 0)
r1, c1 = min(r1 + 1, rate.shape[0]), min(c1 + 1, rate.shape[1])
corner_x, corner_y = np.meshgrid(rate_tf.c + px * np.arange(c0, c1 + 1), rate_tf.f - px * np.arange(r0, r1 + 1))
mesh_x, mesh_y = native_to_map.transform(corner_x, corner_y)
axz.set_facecolor("#eef4fc")  # NISAR pixels: light tint, cell edges on the native 80 m grid
draw_lakes(axz, 1.0, (zx0, zx1, zy0, zy1))  # any lake inside the zoom box (zoom axes are in metres)
for j in range(mesh_x.shape[1]):
    axz.plot(mesh_x[:, j], mesh_y[:, j], color=C_NISAR, lw=0.5, alpha=0.7, zorder=1)
for i in range(mesh_x.shape[0]):
    axz.plot(mesh_x[i, :], mesh_y[i, :], color=C_NISAR, lw=0.5, alpha=0.7, zorder=1)
half = is2.BEAM_SEPARATION_M / 2  # only the centre pair (beams at +/-45 m) reaches a 1 km box
for p in passes:
    if (p.ascending and p.rgt == xo["rgt_asc"]) or (not p.ascending and p.rgt == xo["rgt_desc"]):
        sc.draw_atl06_points(axz, p, (-half, half), (zx0, zx1, zy0, zy1))
axz.set_xlim(zx0, zx1)
axz.set_ylim(zy0, zy1)
axz.set_aspect("equal")
axz.set_xticks([])
axz.set_yticks([])
for s in axz.spines.values():
    s.set_edgecolor(INK)
    s.set_linewidth(1.2)
axz.set_title(f"Sample point: crossover RGT {xo['rgt_asc']:04d} × {xo['rgt_desc']:04d}", loc="left")
axz.plot([zx0 + 60, zx0 + 260], [zy0 + 60, zy0 + 60], color=INK, lw=2, solid_capstyle="butt",
         path_effects=HALO, zorder=5)
axz.text(zx0 + 160, zy0 + 75, "200 m", ha="center", va="bottom", fontsize=7, color=INK, path_effects=HALO, zorder=5)
axz.text(zx0 + 20, zy1 - 20,
         f"NISAR GUNW: {px:.0f} m pixels (outlined; EPSG:3413 grid,\n"
         f"rotated relative to north)\n"
         f"ICESat-2 centre beam pair: 2 beams {is2.BEAM_SEPARATION_M:.0f} m apart;\n"
         f"ATL06 segment every {is2.ATL06_POSTING_M:.0f} m, ~{is2.FOOTPRINT_M:.0f} m footprint",
         ha="left", va="top", fontsize=6.5, color=INK, path_effects=HALO, zorder=5)
for ya in (zy1, zy0):  # connector lines: sample-point square on the map -> left corners of the zoom panel
    fig.add_artist(ConnectionPatch(xyA=(x0 * km, y0 * km), coordsA=ax.transData,
                                   xyB=(zx0, ya), coordsB=axz.transData, color=INK_3, lw=0.6))

handles = [Line2D([], [], color=C_IS2_ASC, lw=1.4, label="ICESat-2 asc: centre beam pair (on RGT)"),
           Line2D([], [], color=C_IS2_ASC, lw=0.6, label=f"ICESat-2 asc: outer beam pairs (±{is2.PAIR_SPACING_M / 1000:.1f} km)"),
           Line2D([], [], color=C_IS2_DESC, lw=1.4, label="ICESat-2 desc: centre beam pair (on RGT)"),
           Line2D([], [], color=C_IS2_DESC, lw=0.6, label=f"ICESat-2 desc: outer beam pairs (±{is2.PAIR_SPACING_M / 1000:.1f} km)"),
           Patch(facecolor="none", edgecolor=C_NISAR, lw=0.9,
                 label="NISAR frame border, ascending (track-frame)"),
           Patch(facecolor="none", edgecolor=C_NISAR, lw=0.9, ls="--",
                 label="NISAR frame border, descending"),
           Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=4.5, label="RGT crossover"),
           Line2D([], [], ls="", marker="o", mfc=C_IS2_ASC, mec="white", ms=4, label="ATL06 segment (20 m posting)"),
           Patch(facecolor=C_LAKE_FILL, edgecolor=C_LAKE_EDGE, lw=0.5, label=f"Lake ≥{LAKE_MIN_KM2:g} km² (USGS NHD)"),
           Patch(facecolor="#eef4fc", edgecolor=C_NISAR, lw=0.6, label="NISAR GUNW 80 m pixel")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.07, "Each ICESat-2 line is a beam pair; its 2 beams (90 m apart) overlap at map scale and are "
         "resolved in the zoom panel.\nNorth-up azimuthal equidistant projection centred on the site. "
         "Lakes: USGS National Hydrography Dataset (NHD) waterbodies.\n" + sc.DEM_ATTRIBUTION, ha="center", va="top", fontsize=6.2, color=INK_2)
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.24)
sc.save_figure(fig, MAP_OUTPUT)


# =============================================================================
# Figure 2 — sampling timeline at the sample point
# Filled markers: acquired (NISAR granule in CMR / ATL06 granule in CMR).
# Open markers: nominal schedule (12-day NISAR repeat, 91-day ICESat-2
# exact repeat) not (yet) present in the archive. Combined rows keep each
# date's own track marker; ICESat-2 rows bracket the gaps between passes.
# =============================================================================
fig = plt.figure(figsize=(7.5, 0.42 * len(rows) + 3.0))
gs = fig.add_gridspec(2, 1, height_ratios=[0.42 * len(rows) + 0.5, 1.5], hspace=0.12)
ax = fig.add_subplot(gs[0])
axd = fig.add_subplot(gs[1], sharex=ax)
yr = np.arange(len(rows))[::-1]
season = [np.datetime64(f"{WINDOW_START[:4]}-{THAW_SEASON[0]}"), np.datetime64(f"{WINDOW_START[:4]}-{THAW_SEASON[1]}")]
for a in (ax, axd):
    a.axvspan(*season, color="#efeee9", zorder=0, lw=0)
ax.text(season[0] + (season[1] - season[0]) / 2, yr[0] + 0.75, "thaw season", ha="center", va="bottom",
        fontsize=7, color=INK_2)
ax.axhline(yr[len(nisar_series)] - 0.5, color=INK_3, lw=0.4)           # NISAR tracks | NISAR combined
ax.axhline(yr[len(nisar_series) + 1] + 0.5, color=INK_3, lw=0.8)       # NISAR | ICESat-2
ax.axhline(yr[-1] + 0.5, color=INK_3, lw=0.4)                          # ICESat-2 RGTs | combined
sc.draw_timeline_rows(ax, rows, yr)
ax.set_xlim(win_start, win_end + np.timedelta64(1, "D"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.grid(axis="x", color="#e4e3df", lw=0.6, zorder=0)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelbottom=False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title(f"Sampling dates at {SITE_NAME} sample point: ICESat-2 crossover RGT {xo['rgt_asc']:04d} × "
             f"{xo['rgt_desc']:04d} ({lat0:.3f}°N, {abs(lon0):.3f}°W)", loc="left")

# Monthly density of NISAR acquisitions at the sample point (all tracks): filled bars = in the
# archive, outlined bars = nominal 12-day schedule on the same tracks. Months after the archive's
# latest acquisition are marked, so unprocessed data isn't mistaken for a seasonal gap.
months = np.arange(np.datetime64(WINDOW_START[:7]), np.datetime64(WINDOW_END[:7]) + 1)
acq_actual = np.concatenate([s["actual"] for s in nisar_series]).astype("datetime64[M]")
acq_nominal = np.concatenate([np.concatenate([s["actual"], s["nominal"]]) for s in nisar_series]).astype("datetime64[M]")
n_actual = np.array([np.sum(acq_actual == mo) for mo in months])
n_nominal = np.array([np.sum(acq_nominal == mo) for mo in months])
centres = months.astype("datetime64[D]") + np.timedelta64(15, "D")
bar_w = np.timedelta64(20, "D")
axd.bar(centres, n_nominal, width=bar_w, facecolor="white", edgecolor=C_NISAR, lw=0.8, zorder=2)
axd.bar(centres, n_actual, width=bar_w, color=C_NISAR, edgecolor="white", lw=0.5, zorder=3)
for c, n in zip(centres, n_actual):
    if n:  # count inside the top of the filled bar, clear of the nominal outline
        axd.text(c, n - 0.3, str(n), ha="center", va="top", fontsize=6.5, color="white", zorder=4)
if archive_end < win_end:
    axd.axvspan(archive_end + np.timedelta64(1, "D"), win_end + np.timedelta64(1, "D"), facecolor="none",
                edgecolor=INK_3, hatch="////", lw=0, zorder=1)
    axd.text(archive_end + (win_end - archive_end) / 2, n_nominal.max() + 1.2,
             f"not yet in archive\n(after {np.datetime_as_string(archive_end)})", ha="center", va="bottom",
             fontsize=6.5, color=INK_2, path_effects=HALO, zorder=5)
axd.set_ylim(0, n_nominal.max() + 4.5)
axd.set_ylabel("NISAR acquisitions\nper month", fontsize=7.5)
axd.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
axd.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
for s in ("top", "right"):
    axd.spines[s].set_visible(False)
axd.set_xlim(win_start, win_end + np.timedelta64(1, "D"))
axd.xaxis.set_major_locator(mdates.MonthLocator())
axd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axd.set_xlabel(WINDOW_START[:4])

legend = [Line2D([], [], ls="", marker="^", color=INK_2, ms=5, label="acquired (in NASA CMR archive)"),
          Line2D([], [], ls="", marker="^", mfc="white", mec=INK_2, ms=5, label="nominal schedule"),
          Line2D([], [], ls="", marker="^", color=INK_3, ms=5, label="ascending"),
          Line2D([], [], ls="", marker="v", color=INK_3, ms=5, label="descending"),
          Patch(facecolor=C_NISAR, edgecolor="white", label="NISAR acquisitions in archive (all tracks)"),
          Patch(facecolor="white", edgecolor=C_NISAR, label="NISAR nominal 12-day schedule (same tracks)")]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.54, -0.01), ncol=3, fontsize=7)
fig.subplots_adjust(left=0.2, right=0.88, top=0.93, bottom=0.14)
sc.save_figure(fig, TIMELINE_OUTPUT)
plt.show()
