# sampling_common.py
# Site-independent helpers shared by the NISAR vs ICESat-2 sampling figure
# scripts (plot_sampling_toolik.py, plot_sampling_greenland.py): CMR granule
# search with a JSON cache, a Copernicus DEM hillshade basemap, NISAR GUNW
# footprint/date parsing, nominal NISAR and ICESat-2 date schedules, and the
# drawing routines for beam tracks, frame borders and timeline rows.
# Importing this module has no side effects other than defining the shared
# plot style (apply_style() sets it).

import json                              # CMR response cache
import os
import re                                # parse granule filenames
import numpy as np
import requests                          # CMR granule search, DEM tiles
import rasterio
from rasterio.merge import merge         # mosaic DEM tiles for the basemap
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import EllipseCollection

import icesat2_tracks as is2             # ICESat-2 nominal RGT helpers


# =============================================================================
# Shared style
# Colours: categorical slots 1-3 of the dataviz reference palette (validated
# all-pairs for colour-vision deficiency).
# =============================================================================
C_NISAR    = "#2a78d6"
C_IS2_ASC  = "#eb6834"
C_IS2_DESC = "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
HALO = [pe.withStroke(linewidth=3, foreground="white")]
LINE_HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
KM = 1e-3

DEM_TILE_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
                "Copernicus_DSM_COG_30_{tile}_DEM/Copernicus_DSM_COG_30_{tile}_DEM.tif")
DEM_ATTRIBUTION = ("Basemap: hillshade of the Copernicus GLO-90 DEM (© DLR e.V. 2010-2014 and © Airbus Defence "
                   "and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA).")


def apply_style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_3, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 9,
                         "axes.titlecolor": INK, "legend.frameon": False})


def save_figure(fig, stem):
    """Write PNG and PDF; skip (with a warning) any file locked by an open viewer."""
    for ext in ("png", "pdf"):
        try:
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
            print(f"Wrote {stem}.{ext}")
        except OSError:  # e.g. PermissionError / EINVAL when a viewer holds the file open
            print(f"WARNING: could not write {stem}.{ext} (file open in another program?)")


def rect_outline(x_lo, y_lo, x_hi, y_hi, n=50):
    """Densified outline of an axis-aligned rectangle (it is curved/rotated in another CRS)."""
    e = np.linspace(0, 1, n)
    xs = np.concatenate([x_lo + e * (x_hi - x_lo), np.full(n, x_hi), x_hi - e * (x_hi - x_lo), np.full(n, x_lo)])
    ys = np.concatenate([np.full(n, y_lo), y_lo + e * (y_hi - y_lo), np.full(n, y_hi), y_hi - e * (y_hi - y_lo)])
    return xs, ys


def geo_bbox_of(to_geo, left, bottom, right, top):
    """lon/lat bounding box (W, S, E, N) of a map-CRS rectangle."""
    elon, elat = to_geo.transform(*rect_outline(left, bottom, right, top))
    return (min(elon), min(elat), max(elon), max(elat))


def north_arrow(ax, x=0.95, y=0.95):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.10), xycoords="axes fraction", ha="center", va="center",
                fontsize=8, fontweight="bold", color=INK, path_effects=HALO, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))


# =============================================================================
# CMR granule search (public metadata), cached to JSON
# =============================================================================
def cmr_search(cache_dir, cache_name, refresh=False, **params):
    """All CMR granule entries for the query, using CMR-Search-After paging. Cached
    per query: a cache file is reused only if it was written for the same parameters."""
    path = os.path.join(cache_dir, cache_name)
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            cached = json.load(f)
        if isinstance(cached, dict) and cached.get("params") == params:
            return cached["entries"]
    entries, headers = [], {}
    while True:
        r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
                         params={**params, "page_size": 2000}, headers=headers, timeout=120)
        r.raise_for_status()
        entries += [{k: e.get(k) for k in ("title", "time_start", "time_end", "polygons")}
                    for e in r.json()["feed"]["entry"]]
        if "CMR-Search-After" not in r.headers:
            break
        headers = {"CMR-Search-After": r.headers["CMR-Search-After"]}
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"params": params, "entries": entries}, f)
    return entries


# =============================================================================
# Basemap: hillshade of the Copernicus GLO-90 DEM (public AWS open data, 1x1
# degree COG tiles, cached locally), mosaicked and reprojected onto the map
# grid. Tiles over open ocean don't exist and are skipped.
# =============================================================================
def dem_tile_name(lat, lon):
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"


def load_hillshade(geo_bbox, map_crs, left, right, bottom, top, cache_dir, res_m=100.0, z_factor=2.0):
    """Hillshade (0-1) on a map-CRS grid covering [left, right] x [bottom, top], and its imshow extent."""
    os.makedirs(cache_dir, exist_ok=True)
    srcs = []
    for lat in range(int(np.floor(geo_bbox[1])), int(np.floor(geo_bbox[3])) + 1):
        for lon in range(int(np.floor(geo_bbox[0])), int(np.floor(geo_bbox[2])) + 1):
            tile = dem_tile_name(lat, lon)
            path = os.path.join(cache_dir, f"{tile}.tif")
            if not os.path.exists(path):
                r = requests.get(DEM_TILE_URL.format(tile=tile), timeout=120)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            srcs.append(rasterio.open(path))
    mosaic, mosaic_tf = merge(srcs)
    for s in srcs:
        s.close()
    n_cols = int(np.ceil((right - left) / res_m))
    n_rows = int(np.ceil((top - bottom) / res_m))
    dem = np.full((n_rows, n_cols), np.nan, np.float32)
    reproject(mosaic[0].astype(np.float32), dem, src_transform=mosaic_tf, src_crs="EPSG:4326",
              dst_transform=rasterio.transform.from_origin(left, top, res_m, res_m),
              dst_crs=map_crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
    # Lambertian shading, light from the NW (azimuth 315 deg) at 45 deg elevation
    dz_drow, dz_dx = np.gradient(dem * z_factor, res_m)
    dz_dy = -dz_drow                                  # rows run southward
    az, alt = np.radians(315.0), np.radians(45.0)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])  # east, north, up
    shade = (-dz_dx * light[0] - dz_dy * light[1] + light[2]) / np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1)
    return np.clip(shade, 0, 1), (left, left + n_cols * res_m, top - n_rows * res_m, top)


def draw_hillshade(ax, hillshade, extent):
    ax.imshow(hillshade, cmap="gray", vmin=-0.6, vmax=1.15, interpolation="bilinear",  # compressed gray range:
              extent=[v * KM for v in extent], origin="upper", zorder=0)              # relief stays recessive


# =============================================================================
# NISAR GUNW granules: frames and acquisitions at a sample point
# Each GUNW granule covers one frame of one track and contributes two
# acquisitions (reference and secondary date). Footprints are CMR's 4-corner
# polygons (straight edges in lon/lat), so edges are densified before
# projecting.
# =============================================================================
GUNW_RE = re.compile(r"NISAR_L2_PR_GUNW_\d{3}_(\d{3})_([AD])_(\d{3})_\d{3}_\d{4}_\w{2}_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T")


def ymd(d):
    return np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}")


def footprint_path(polygon_str, to_map):
    latlon = np.array(polygon_str.split(), float).reshape(-1, 2)
    t = np.linspace(0, 1, 40, endpoint=False)
    dense = np.concatenate([a + t[:, None] * (b - a) for a, b in zip(latlon[:-1], latlon[1:])])
    fx, fy = to_map.transform(dense[:, 1], dense[:, 0])
    return Path(np.column_stack([np.append(fx, fx[0]), np.append(fy, fy[0])]), closed=True)


def scan_nisar(entries, to_map, map_box, x0, y0):
    """
    Returns (frames, point_dates, archive_end):
      frames      — (track, direction, frame) -> footprint path (latest granule's polygon) for frames
                    intersecting map_box;
      point_dates — (track, direction) -> set of acquisition dates whose footprint covers (x0, y0);
      archive_end — latest archived acquisition date in the query.
    """
    frames, point_dates = {}, {}
    archive_end = np.datetime64("1970-01-01")
    for e in sorted(entries, key=lambda e: e["time_start"]):
        m = GUNW_RE.match(e["title"])
        if not m or not e.get("polygons"):
            continue
        track, direction, frame = int(m.group(1)), m.group(2), int(m.group(3))
        archive_end = max(archive_end, ymd(m.group(5)))  # latest archived acquisition
        path = footprint_path(e["polygons"][0][0], to_map)
        if path.intersects_bbox(map_box):
            frames[(track, direction, frame)] = path
        if path.contains_point((x0, y0)):
            for d in (m.group(4), m.group(5)):
                point_dates.setdefault((track, direction), set()).add(ymd(d))
    return frames, point_dates, archive_end


# =============================================================================
# Date schedules
# NISAR: actual (CMR) + nominal 12-day grid anchored on the actual dates.
# ICESat-2: nominal 91-day repeat; a nominal pass counts as acquired if an
# ATL06 granule on that RGT starts within a day of it (granules start a few
# minutes before reaching the AOI).
# =============================================================================
def nisar_dates(dates, win_start, win_end, repeat_days=12):
    step = np.timedelta64(repeat_days, "D")
    actual = np.array(sorted(dates))
    k = np.arange(np.floor((win_start - actual[0]) / step), np.ceil((win_end - actual[0]) / step) + 1)
    grid = actual[0] + k.astype(int) * step
    grid = grid[(grid >= win_start) & (grid <= win_end)]
    actual = actual[(actual >= win_start) & (actual <= win_end)]
    nominal = np.array([g for g in grid if not len(actual) or np.min(np.abs(actual - g)) > np.timedelta64(1, "D")],
                       dtype="datetime64[D]")
    return actual, nominal


ATL06_RE = re.compile(r"ATL06_\d{14}_(\d{4})\d{4}_")


def atl06_starts_by_rgt(entries):
    starts = {}
    for e in entries:
        m = ATL06_RE.match(e["title"])
        if m:
            starts.setdefault(int(m.group(1)), []).append(np.datetime64(e["time_start"][:19], "ms"))
    return starts


def is2_dates(rgt, t_pass, atl06_starts, window_start, window_end):
    nominal = is2.pass_dates(t_pass, window_start, f"{window_end}T23:59:59")
    starts = np.array(atl06_starts.get(rgt, []), dtype="datetime64[ms]")
    got = np.array([len(starts) > 0 and np.min(np.abs(starts - t)) < np.timedelta64(1, "D") for t in nominal], bool)
    return nominal[got], nominal[~got]


# =============================================================================
# Map drawing
# =============================================================================
def draw_frame_borders(ax, frames, left, right, bottom, top, avoid):
    """
    NISAR frame borders (outlines; solid = ascending, dashed = descending) with labels placed
    on each outline inside the map, greedily at the candidate point farthest from labels already
    placed and from the `avoid` points. Returns the names of frames whose outline never enters
    the map (they cover it entirely), for a corner note.
    """
    for (track, direction, frame), path in frames.items():
        ls = "-" if direction == "A" else (0, (4, 2))
        ax.add_patch(PathPatch(Path(path.vertices * KM, path.codes), facecolor="none", edgecolor=C_NISAR, lw=0.9,
                               ls=ls, path_effects=LINE_HALO, zorder=4))
    covering, placed = [], [np.asarray(q, float) for q in avoid]
    for (track, direction, frame), path in sorted(frames.items()):
        v = path.vertices
        inside = (v[:, 0] > left + 4e3) & (v[:, 0] < right - 4e3) & (v[:, 1] > bottom + 2e3) & (v[:, 1] < top - 2e3)
        name = f"{track:03d}{direction}-{frame:03d}"
        if not inside.any():
            covering.append(name)
            continue
        cand = v[inside]
        clearance = np.min([np.hypot(*(cand - q).T) for q in placed], axis=0)
        lx, ly = cand[np.argmax(clearance)]
        placed.append(np.array([lx, ly]))
        ax.text(lx * KM, ly * KM, name, fontsize=6, color=C_NISAR, ha="center", va="center",
                path_effects=HALO, zorder=7)
    return covering


def draw_beams(ax, passes, scale=KM, centre_lw=1.1, outer_lw=0.45, include_weak=True):
    """ICESat-2: all six beams per pass; centre pair (on the RGT) thicker than the two outer pairs."""
    for p in passes:
        colour = C_IS2_ASC if p.ascending else C_IS2_DESC
        for off, (bx, by) in is2.beam_tracks(p, include_weak=include_weak).items():
            centre = abs(off) < is2.PAIR_SPACING_M / 2
            ax.plot(bx * scale, by * scale, color=colour, lw=centre_lw if centre else outer_lw,
                    solid_capstyle="butt", zorder=5)


def draw_atl06_points(ax, p, offsets, box, margin=50.0):
    """Beams at the given cross-track offsets of pass p, with ATL06 segment centres (20 m) and footprints."""
    zx0, zx1, zy0, zy1 = box
    colour = C_IS2_ASC if p.ascending else C_IS2_DESC
    for off, (bx, by) in is2.beam_tracks(p).items():
        if not any(abs(off - o) < 1.0 for o in offsets):
            continue
        sx, sy, _ = is2.resample(bx, by, p.t, is2.ATL06_POSTING_M)
        keep = (sx > zx0 - margin) & (sx < zx1 + margin) & (sy > zy0 - margin) & (sy < zy1 + margin)
        ax.plot(sx[keep], sy[keep], color=colour, lw=0.8, path_effects=LINE_HALO, zorder=2)
        ax.add_collection(EllipseCollection(
            np.full(keep.sum(), is2.FOOTPRINT_M), np.full(keep.sum(), is2.FOOTPRINT_M), np.zeros(keep.sum()),
            units="xy", offsets=np.column_stack([sx[keep], sy[keep]]), offset_transform=ax.transData,
            facecolor=colour, edgecolor="white", linewidth=0.4, zorder=3))


# =============================================================================
# Timeline drawing
# Each row: {"label", "series": [series...], "brackets": bool, "combined": bool,
# "bracket_labels": bool (default True)}; each series: {"actual", "nominal",
# "colour", "marker"}. Filled markers = acquired, open = nominal schedule.
# =============================================================================
def all_dates(series):
    return np.unique(np.concatenate([np.concatenate([s["actual"], s["nominal"]]) for s in series])
                     .astype("datetime64[D]"))


def draw_timeline_rows(ax, rows, yr):
    for y, r in zip(yr, rows):
        dates = all_dates(r["series"])
        gaps = np.diff(dates).astype(int)
        if r["brackets"]:  # bracket each revisit gap
            for d1, d2, g in zip(dates[:-1], dates[1:], gaps):
                ax.annotate("", xy=(d2, y + 0.22), xytext=(d1, y + 0.22),
                            arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color=INK_3, lw=0.7))
                if r.get("bracket_labels", True):
                    ax.text(d1 + (d2 - d1) / 2, y + 0.27, f"{g} d", ha="center", va="bottom", fontsize=6.5,
                            color=INK_2)
        size = 16 if r.get("combined") and not r["brackets"] else 26  # dense NISAR combined row: smaller markers
        for s in r["series"]:
            ax.scatter(s["actual"], np.full(len(s["actual"]), y), marker=s["marker"], s=size, color=s["colour"],
                       edgecolor="white", lw=0.5, zorder=3)
            ax.scatter(s["nominal"], np.full(len(s["nominal"]), y), marker=s["marker"], s=size * 0.85,
                       facecolor="white", edgecolor=s["colour"], lw=0.8, zorder=3)
        ax.text(1.01, y, f"{np.median(gaps):.0f} d", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7.5, color=INK)
        if r.get("combined"):  # also report the longest gap between passes
            ax.text(1.065, y, f"(max {gaps.max()} d)", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6.5, color=INK_2)

    ax.text(1.01, yr[0] + 0.75, "median\nrevisit", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7,
            color=INK_2)
    ax.set_yticks(yr)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    for lbl, r in zip(ax.get_yticklabels(), rows):
        if r.get("combined"):
            lbl.set_fontweight("bold")
    ax.set_ylim(yr[-1] - 0.6, yr[0] + 1.2)
