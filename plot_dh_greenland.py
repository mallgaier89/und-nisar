# plot_dh_greenland.py
# Elevation-change time series at the Greenland study-area sample point (the
# ICESat-2 crossover chosen by plot_sampling_greenland.py, east of Point 660),
# comparing:
#   - NISAR: per-pair "apparent vertical" displacement from the GUNW unwrapped
#     phase, processed the same way as the InSAR pipeline (coherence mask,
#     phase -> LOS -> vertical = LOS / cos(incidence)), referenced to stable
#     ice-free bedrock near Point 660, and accumulated along each track's
#     sequential pair chain — for the 80 m pixel at the crossover and for the
#     mean over the ATL15 1 km cell containing it;
#   - ICESat-2 ATL15: gridded height change (delta_h, relative to the ATL14
#     2020.0 surface) at the 1 km (3-month) and 2.5 km (monthly) cells.
# CAVEAT: on flowing ice the LOS signal is dominated by horizontal ice motion,
# so the NISAR "vertical" here is not surface-elevation change (see figure note).
#
# NISAR GUNW granules (~2 GB each) are not downloaded: only small windows are
# streamed with earthaccess (Earthdata login), and per-pair results are cached
# in ./data/nisar_dh_point660.json. ATL15 cell values are cached in
# ./data/atl15_dh_point660.json.

import glob
import json
import os
import numpy as np
import h5py
import rasterio
from rasterio.merge import merge
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sampling_common as sc             # shared CMR / style helpers
import nisar_corrections as nc           # GUNW ionospheric / tropospheric corrections (shared with the pipeline)
from sampling_common import C_NISAR, INK, INK_2, INK_3, HALO


# =============================================================================
# Parameters
# =============================================================================
SAMPLE_POINT_JSON = "./data/Point660_sample_point.json"   # written by plot_sampling_greenland.py
NISAR_SHORT_NAME  = "NISAR_L2_GUNW_PROVISIONAL_V1"
NISAR_TEMPORAL    = "2025-06-01T00:00:00Z,2027-01-01T00:00:00Z"
CMR_CACHE_DIR     = "./data/cmr_cache"
NISAR_CACHE       = "./data/nisar_dh_point660.json"
ATL15_CACHE       = "./data/atl15_dh_point660.json"
GIMP_DIR          = "./data/gimp"                         # GIMP 2020 ice-mask tiles cached by plot_sampling_greenland.py
GIMP_PATTERN      = "GimpIceMask_15m_2020_tile*_v1.2.tif"
REF_LON, REF_LAT  = -50.20, 67.15     # centre of the bedrock reference region, ~7 km west of Point 660
REF_HALF_M        = 3000.0            # half-width of the reference region (m)
REF_COH_MIN       = 0.30              # reference pixels must be at least this coherent
GRID_CRS          = "EPSG:3413"       # GUNW and ATL15 grid over Greenland
MELT_SEASON       = ("06-01", "08-31")  # MM-DD span shaded each year
CLIM_YEARS        = (2019, 2025)      # ATL15 years used for the seasonal reference rates
PAIR_DAYS         = 12                # NISAR repeat: ATL15 reference rates are scaled to this pair length
FIGURE_OUTPUT     = "./data/Point660 dh_timeseries"
TRACK_MARKERS     = {"A": ["^", "s"], "D": ["v", "D"]}  # per direction, in track order

# Mirrored from process_displacement.py (the InSAR pipeline) — keep in sync
COHERENCE_MIN     = 0.15              # pixels below this coherence magnitude are masked
# Mirrored from download_nisar.py — keep in sync
APPLY_IONOSPHERE_CORRECTION  = False  # subtract the GUNW ionospherePhaseScreen; off for now — unreliable on the ice (LIMITATIONS.md)
APPLY_TROPOSPHERE_CORRECTION = True   # subtract the GUNW wet + hydrostatic tropospheric phase screens
CORRECTIONS = {"ionosphere": APPLY_IONOSPHERE_CORRECTION, "troposphere": APPLY_TROPOSPHERE_CORRECTION}
_applied = [k for k, v in CORRECTIONS.items() if v]
CORRECTION_NOTE = (f"GUNW {' + '.join(_applied)} phase screen{'s' if len(_applied) > 1 else ''} applied"
                   if _applied else "no GUNW phase-screen corrections applied")
if not APPLY_IONOSPHERE_CORRECTION and APPLY_TROPOSPHERE_CORRECTION:
    CORRECTION_NOTE += " (ionosphere screen not applied: unreliable on the ice)"
SIGN_CONVENTION   = -1.0              # phase -> LOS sign, as in process_displacement.py
# Mirrored from download_nisar.py (GUNW HDF5 layout)
C_LIGHT           = 299_792_458.0
G = "/science/LSAR/GUNW/"
PHASE_PATH        = G + "grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase"
COHERENCE_PATH    = G + "grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude"
CONNCOMP_PATH     = G + "grids/frequencyA/unwrappedInterferogram/HH/connectedComponents"
X_COORD_PATH      = G + "grids/frequencyA/unwrappedInterferogram/HH/xCoordinates"
Y_COORD_PATH      = G + "grids/frequencyA/unwrappedInterferogram/HH/yCoordinates"
CENTER_FREQ_PATH  = G + "grids/frequencyA/centerFrequency"
RADARGRID_INC_PATH     = G + "metadata/radarGrid/incidenceAngle"
RADARGRID_X_PATH       = G + "metadata/radarGrid/xCoordinates"
RADARGRID_Y_PATH       = G + "metadata/radarGrid/yCoordinates"
RADARGRID_HEIGHTS_PATH = G + "metadata/radarGrid/heightAboveEllipsoid"
REF_TERRAIN_HEIGHT_PATH = G + "metadata/processingInformation/parameters/reference/referenceTerrainHeight"


# =============================================================================
# Sample point and windows (EPSG:3413)
# =============================================================================
if not os.path.exists(SAMPLE_POINT_JSON):
    raise SystemExit(f"{SAMPLE_POINT_JSON} not found — run plot_sampling_greenland.py first")
with open(SAMPLE_POINT_JSON) as f:
    sp = json.load(f)
xp, yp = sp["x_3413"], sp["y_3413"]
cell = sp["atl15_cells"]["3mo_1km"]                      # 1 km ATL15 cell containing the point
cell_box = (cell["x"] - cell["cell_m"] / 2, cell["x"] + cell["cell_m"] / 2,
            cell["y"] - cell["cell_m"] / 2, cell["y"] + cell["cell_m"] / 2)
to_grid = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
xr, yr = to_grid.transform(REF_LON, REF_LAT)
ref_box = (xr - REF_HALF_M, xr + REF_HALF_M, yr - REF_HALF_M, yr + REF_HALF_M)
print(f"Sample point {sp['lat']:.4f}N {sp['lon']:.4f}E; ATL15 1 km cell centre ({cell['x']:.0f}, {cell['y']:.0f})")


def gimp_ice_at(xs, ys):
    """GIMP 2020 land-ice flag (True = ice) at EPSG:3413 points, from the cached 15 m tiles."""
    srcs = [rasterio.open(p) for p in sorted(glob.glob(os.path.join(GIMP_DIR, GIMP_PATTERN)))]
    if not srcs:
        raise SystemExit(f"No GIMP tiles in {GIMP_DIR} — run plot_sampling_greenland.py first")
    try:
        mosaic, tf = merge(srcs, bounds=(xs.min() - 30, ys.min() - 30, xs.max() + 30, ys.max() + 30))
    finally:
        for s in srcs:
            s.close()
    rows, cols = rasterio.transform.rowcol(tf, xs.ravel(), ys.ravel())
    return (mosaic[0][np.asarray(rows), np.asarray(cols)] == 1).reshape(xs.shape)


# =============================================================================
# NISAR: per-pair values from streamed GUNW windows (cached per granule)
# =============================================================================
def window(coords, lo, hi):
    """Index slice of a monotonic coordinate vector covering [lo, hi] (pixel centres)."""
    idx = np.flatnonzero((coords >= lo) & (coords <= hi))
    return slice(int(idx.min()), int(idx.max()) + 1) if idx.size else None


def process_granule(h):
    """Pipeline-equivalent per-pair numbers for the target pixel, its ATL15 1 km cell, and the reference."""
    x, y = h[X_COORD_PATH][:], h[Y_COORD_PATH][:]
    tc, tr = window(x, cell_box[0], cell_box[1]), window(y, cell_box[2], cell_box[3])
    rc, rr = window(x, ref_box[0], ref_box[1]), window(y, ref_box[2], ref_box[3])
    if None in (tc, tr, rc, rr):
        return {"status": "outside grid"}
    read = lambda path, r, c: h[path][r, c].astype(np.float64)
    ph_t, coh_t, cc_t = read(PHASE_PATH, tr, tc), read(COHERENCE_PATH, tr, tc), read(CONNCOMP_PATH, tr, tc)
    ph_r, coh_r, cc_r = read(PHASE_PATH, rr, rc), read(COHERENCE_PATH, rr, rc), read(CONNCOMP_PATH, rr, rc)

    # reference: coherent, ice-free pixels in the reference region's dominant (non-zero) connected component
    gx, gy = np.meshgrid(x[rc], y[rr])
    ref_ok = (~gimp_ice_at(gx, gy)) & np.isfinite(ph_r) & (coh_r >= REF_COH_MIN) & (cc_r > 0)
    if ref_ok.sum() < 50:
        return {"status": "no coherent reference", "ref_n": int(ref_ok.sum())}
    labels, counts = np.unique(cc_r[ref_ok], return_counts=True)
    ref_cc = labels[np.argmax(counts)]
    ref_ok &= cc_r == ref_cc

    # ionospheric / tropospheric correction of both windows (nisar_corrections.py, as in download_nisar.py)
    crs = f"EPSG:{int(h[nc.PROJ_PATH][()])}"
    screens = {}
    for name, r_, c_ in (("t", tr, tc), ("r", rr, rc)):
        xx, yy = np.meshgrid(x[c_], y[r_])
        screens[name] = (nc.ionosphere_screen(h, r_, c_),
                         nc.troposphere_screen(h, xx, yy, nc.terrain_height(xx, yy, crs)))
    (iono_t, tropo_t), (iono_r, tropo_r) = screens["t"], screens["r"]
    # sign check on stable bedrock: subtracting a correct screen should reduce the phase scatter there
    scatter = lambda a: float(np.std(a[ref_ok]))
    sign_check = {"raw": scatter(ph_r), "minus_iono": scatter(ph_r - iono_r), "plus_iono": scatter(ph_r + iono_r),
                  "minus_tropo": scatter(ph_r - tropo_r), "plus_tropo": scatter(ph_r + tropo_r)}
    ph_t = ph_t - APPLY_IONOSPHERE_CORRECTION * iono_t - APPLY_TROPOSPHERE_CORRECTION * tropo_t
    ph_r = ph_r - APPLY_IONOSPHERE_CORRECTION * iono_r - APPLY_TROPOSPHERE_CORRECTION * tropo_r
    ref_phase = float(np.median(ph_r[ref_ok]))

    # incidence angle at the pixel: coarse radarGrid, layer nearest the median reference terrain height
    heights = h[RADARGRID_HEIGHTS_PATH][:]
    k = int(np.argmin(np.abs(heights - np.median(h[REF_TERRAIN_HEIGHT_PATH][:]))))
    inc = RegularGridInterpolator((h[RADARGRID_Y_PATH][:], h[RADARGRID_X_PATH][:]),
                                  h[RADARGRID_INC_PATH][k, :, :], bounds_error=False, fill_value=None)
    incidence = float(inc((yp, xp)))
    wavelength = C_LIGHT / float(h[CENTER_FREQ_PATH][()])

    # phase -> referenced LOS -> "apparent vertical" (process_displacement.py), valid-pixel mask
    valid = np.isfinite(ph_t) & (coh_t >= COHERENCE_MIN) & (cc_t == ref_cc)
    dv = SIGN_CONVENTION * wavelength / (4 * np.pi) * (ph_t - ref_phase) / np.cos(np.radians(incidence))
    pc, pr = int(np.argmin(np.abs(x[tc] - xp))), int(np.argmin(np.abs(y[tr] - yp)))
    return {
        "status": "ok", "incidence_deg": incidence, "wavelength_m": wavelength,
        "pixel_dv_m": float(dv[pr, pc]) if valid[pr, pc] else None,
        "pixel_coh": float(coh_t[pr, pc]), "pixel_cc": int(cc_t[pr, pc]),
        "cell_dv_m": float(np.mean(dv[valid])) if valid.any() else None,
        "cell_n_valid": int(valid.sum()), "cell_n": int(np.isfinite(ph_t).sum()),
        "cell_coh_median": float(np.nanmedian(coh_t)),
        "ref_n": int(ref_ok.sum()), "ref_cc": int(ref_cc), "ref_coh_median": float(np.median(coh_r[ref_ok])),
        "ref_phase_rad": ref_phase,
        "corrections": CORRECTIONS,
        # correction at the target relative to the reference (radians); what referencing leaves of each screen
        "iono_t_minus_r_rad": float(np.nanmedian(iono_t) - np.median(iono_r[ref_ok])),
        "tropo_t_minus_r_rad": float(np.nanmedian(tropo_t) - np.median(tropo_r[ref_ok])),
        "ref_scatter_rad": sign_check,
    }


entries = sc.cmr_search(CMR_CACHE_DIR, "Point660_nisar_gunw_point.json", short_name=NISAR_SHORT_NAME,
                        point=f"{sp['lon']:.5f},{sp['lat']:.5f}", temporal=NISAR_TEMPORAL)
latest = {}  # keep the highest product version per (track, direction, frame, reference, secondary)
for e in entries:
    m = sc.GUNW_RE.match(e["title"])
    if m:
        key = (int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5))
        if key not in latest or e["title"] > latest[key]:
            latest[key] = e["title"]
print(f"{len(latest)} NISAR GUNW granules cover the sample point")

cache = {}
if os.path.exists(NISAR_CACHE):
    with open(NISAR_CACHE) as f:
        cache = json.load(f)
todo = [t for t in latest.values() if cache.get(t, {}).get("corrections") != CORRECTIONS]  # new or other settings
if todo:
    import earthaccess
    earthaccess.login(strategy="netrc")
    for i, title in enumerate(sorted(todo)):
        granules = earthaccess.search_data(short_name=NISAR_SHORT_NAME, granule_name=title, count=1)
        fh = earthaccess.open(granules)[0]
        try:
            with h5py.File(fh, "r") as h:
                cache[title] = process_granule(h)
        except Exception as err:  # network / format problem: record and move on
            cache[title] = {"status": f"error: {err}"}
        finally:
            fh.close()
        r = cache[title]
        print(f"  [{i + 1}/{len(todo)}] {title[17:40]}… {r['status']}"
              + (f", pixel coh {r['pixel_coh']:.2f}, cell valid {r['cell_n_valid']}/{r['cell_n']}, "
                 f"ref n {r['ref_n']} coh {r['ref_coh_median']:.2f}" if r["status"] == "ok" else ""))
        with open(NISAR_CACHE, "w") as f:  # save after every granule so an interrupted run resumes
            json.dump(cache, f, indent=1)

# one row per (track, pair): prefer the frame with the most valid cell pixels where frames overlap
pairs = {}
for (track, direction, frame, ref, sec), title in latest.items():
    r = cache.get(title, {})
    if r.get("status") != "ok":
        continue
    key = (track, direction, ref, sec)
    if key not in pairs or r["cell_n"] > pairs[key]["cell_n"]:
        pairs[key] = {**r, "frame": frame, "title": title}
print(f"{len(pairs)} usable pairs")
for (track, direction, ref, sec), r in sorted(pairs.items()):
    fmt = lambda v: f"{100 * v:+6.1f} cm" if v is not None else "   masked"
    print(f"  T{track:03d}{direction} F{r['frame']:03d} {ref}-{sec}: pixel {fmt(r['pixel_dv_m'])} "
          f"(coh {r['pixel_coh']:.2f}, cc {r['pixel_cc']}), cell {fmt(r['cell_dv_m'])} "
          f"({r['cell_n_valid']}/{r['cell_n']} valid, med coh {r['cell_coh_median']:.2f}), "
          f"inc {r['incidence_deg']:.1f}, ref coh {r['ref_coh_median']:.2f}")
    to_cm = 100 * r["wavelength_m"] / (4 * np.pi) / np.cos(np.radians(r["incidence_deg"]))  # rad -> cm, apparent vertical
    print(f"      correction target-minus-reference: ionosphere {r['iono_t_minus_r_rad'] * to_cm:+5.1f} cm, "
          f"troposphere {r['tropo_t_minus_r_rad'] * to_cm:+5.1f} cm")

# Sign check: on stable bedrock, subtracting the screens should reduce the phase scatter in most pairs
sc_ = [r["ref_scatter_rad"] for r in pairs.values()]
better = {k: sum(s[f"minus_{k}"] < s[f"plus_{k}"] for s in sc_) for k in ("iono", "tropo")}
print(f"Sign check on bedrock reference ({len(sc_)} pairs): subtracting reduces scatter vs adding in "
      f"{better['iono']} pairs (ionosphere), {better['tropo']} pairs (troposphere); median scatter raw "
      f"{np.median([s['raw'] for s in sc_]):.2f} rad, minus iono {np.median([s['minus_iono'] for s in sc_]):.2f}, "
      f"minus tropo {np.median([s['minus_tropo'] for s in sc_]):.2f}")
for k in ("iono", "tropo"):
    if better[k] < len(sc_) / 2:
        raise SystemExit(f"Sign check failed for {k}: subtracting the screen does not reduce bedrock scatter — "
                         "check the sign convention in nisar_corrections.py before using these results")


# =============================================================================
# ATL15: delta_h (time, y, x) and its sigma at the cells containing the point,
# streamed from the ATL15 files (only the cell's time series is read), cached.
# delta_h is relative to the ATL14 2020.0 surface; sigma is inf at that epoch.
# =============================================================================
atl15 = {}
if os.path.exists(ATL15_CACHE):
    with open(ATL15_CACHE) as f:
        atl15 = json.load(f)
for key, c in sp["atl15_cells"].items():
    if atl15.get(key, {}).get("title") == c["title"]:
        continue
    import earthaccess
    earthaccess.login(strategy="netrc")
    granules = earthaccess.search_data(short_name="ATL15", granule_name=c["title"] + "*", count=1)
    fh = earthaccess.open(granules)[0]
    try:
        with h5py.File(fh, "r") as h:
            d = h["delta_h"]
            i = int(np.argmin(np.abs(d["x"][:] - c["x"])))
            j = int(np.argmin(np.abs(d["y"][:] - c["y"])))
            series = lambda name: d[name][:, j, i].astype(float) if name in d else None
            dh, sigma, count = series("delta_h"), series("delta_h_sigma"), series("data_count")
            t = d["time"][:]
    finally:
        fh.close()
    bad = dh > 1e30                                            # _FillValue
    clean = lambda a: [None if (b or not np.isfinite(v)) else float(v) for v, b in zip(a, bad)] if a is not None else None
    atl15[key] = {"title": c["title"], "cell_m": c["cell_m"], "x": c["x"], "y": c["y"],
                  "epochs": [str(np.datetime64("2018-01-01") + np.timedelta64(int(round(v)), "D")) for v in t],
                  "delta_h_m": clean(dh), "sigma_m": clean(np.where(~(sigma < 1e30), 0.0, sigma)),
                  "data_count": clean(count)}
    with open(ATL15_CACHE, "w") as f:
        json.dump(atl15, f, indent=1)
for key, a in atl15.items():
    dh = np.array([np.nan if v is None else v for v in a["delta_h_m"]])
    print(f"ATL15 {key} cell ({a['x']:.0f}, {a['y']:.0f}): {len(dh)} epochs {a['epochs'][0]} - {a['epochs'][-1]}, "
          f"dh {np.nanmin(dh):+.2f} to {np.nanmax(dh):+.2f} m")


# =============================================================================
# ATL15 seasonal reference rates, scaled to one NISAR pair (PAIR_DAYS): the
# mean +/- SD over CLIM_YEARS of the height change between two seasonal
# epochs. NISAR pairs only survive coherence masking in Sep-Oct, so two
# references are shown: the 1 km 3-month Jul->Oct change (whole melt season)
# and the 2.5 km monthly Sep->Nov change (same window as the valid pairs).
# =============================================================================
def atl15_arrays(key):
    a = atl15[key]
    as_arr = lambda v: np.array([np.nan if x is None else x for x in v])
    sigma = as_arr(a["sigma_m"])
    sigma[~(sigma < 1e30)] = 0.0  # fill value at the ATL14 reference epoch (dh = 0 by definition)
    return np.array(a["epochs"], dtype="datetime64[D]"), as_arr(a["delta_h_m"]), sigma


def seasonal_rate(key, start_md, end_md):
    epochs, dh, _ = atl15_arrays(key)
    rates = []
    for year in range(CLIM_YEARS[0], CLIM_YEARS[1] + 1):
        i = int(np.argmin(np.abs(epochs - np.datetime64(f"{year}-{start_md}"))))
        j = int(np.argmin(np.abs(epochs - np.datetime64(f"{year}-{end_md}"))))
        rates.append((dh[j] - dh[i]) * PAIR_DAYS / (epochs[j] - epochs[i]).astype(int))
    rates = np.array(rates)
    return float(np.nanmean(rates)), float(np.nanstd(rates))


ref_rates = {"ATL15 1 km, Jul→Oct": seasonal_rate("3mo_1km", "07-01", "10-01"),
             "ATL15 2.5 km, Sep→Nov": seasonal_rate("1mo_2.5km", "09-01", "11-01")}
for name, (mu, sd) in ref_rates.items():
    print(f"{name.replace('→', '->')} {CLIM_YEARS[0]}-{CLIM_YEARS[1]}: {100 * mu:+.1f} +/- {100 * sd:.1f} cm "
          f"per {PAIR_DAYS} days")


# =============================================================================
# Figure — (a) ATL15 history, (b) NISAR per-pair "apparent vertical"
# displacement vs ATL15 seasonal rates, (c) per-pair coherence
# =============================================================================
sc.apply_style()
ymd = lambda s: np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:]}")
tracks = sorted({(t, d) for t, d, _, _ in pairs}, key=lambda k: (k[1], k[0]))
marker = {}
for t, d in tracks:
    marker[(t, d)] = TRACK_MARKERS[d][sum(1 for k in marker if k[1] == d)]
nisar_start = min(ymd(k[2]) for k in pairs)
nisar_end = max(ymd(k[3]) for k in pairs)

def track_handles(ms=6):
    return [Line2D([], [], ls="", marker=marker[k], color=C_NISAR, mec="white", ms=ms,
                   label=f"NISAR track {k[0]:03d} {'asc' if k[1] == 'A' else 'desc'}") for k in tracks]


SOURCES_ATL15 = f"ATL15: {atl15['3mo_1km']['title']}, {atl15['1mo_2.5km']['title']}."
SOURCES_NISAR = (f"NISAR: {NISAR_SHORT_NAME} unwrapped phase, coherence-masked and \n{CORRECTION_NOTE} "
                 "(nisar_corrections.py), converted as in process_displacement.py;\n"
                 "pairs with valid pixels in the bedrock reference's connected component only.")
SAMPLE_NOTE = (f"Sample point: ICESat-2 crossover RGT {sp['rgt_asc']:04d} × {sp['rgt_desc']:04d} east of Point 660; "
               "NISAR 1 km cell = ATL15 1 km cell containing it.")
melt_patch = Patch(facecolor="#efeee9", label="Melt season (Jun–Aug)")


def draw_panel_a(ax, standalone=False):
    """ATL15 delta_h history at the cells. Standalone: own legend incl. melt season and NISAR period."""
    for year in range(2019, 2027):
        ax.axvspan(np.datetime64(f"{year}-{MELT_SEASON[0]}"), np.datetime64(f"{year}-{MELT_SEASON[1]}"),
                   color="#efeee9", lw=0, zorder=0)
    ax.axvspan(nisar_start, nisar_end, color=C_NISAR, alpha=0.12, lw=0, zorder=0)
    ax.text(nisar_start + (nisar_end - nisar_start) / 2, 0.97, "NISAR" if standalone else "NISAR\n(b)",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color=C_NISAR)
    for key, colour, lw, ms, label in (("1mo_2.5km", INK_3, 0.8, 0, "ATL15 2.5 km cell, monthly"),
                                       ("3mo_1km", INK, 1.4, 3.5, "ATL15 1 km cell, 3-month")):
        epochs, dh, sigma = atl15_arrays(key)
        ax.fill_between(epochs, dh - sigma, dh + sigma, color=colour, alpha=0.18, lw=0, zorder=1)
        ax.plot(epochs, dh, color=colour, lw=lw, marker="o" if ms else None, ms=ms, mec="white", mew=0.5,
                label=label, zorder=2)
    ax.axhline(0, color=INK_3, lw=0.5, zorder=0.5)
    ax.set_xlim(np.datetime64("2019-01-01"), np.datetime64("2026-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Height change\nvs ATL14 2020.0 (m)")
    ax.set_title(f"{'' if standalone else '(a) '}ICESat-2 ATL15 height change at the sample-point cell "
                 f"({sp['lat']:.3f}°N, {abs(sp['lon']):.3f}°W)", loc="left")
    if standalone:
        handles, _ = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color=INK_3, lw=6, alpha=0.3, label="±1σ (ATL15 delta_h_sigma)"), melt_patch,
                    Patch(facecolor=C_NISAR, alpha=0.12, label="NISAR observation period")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=6.8)
    else:
        ax.legend(loc="upper center", fontsize=6.8, ncol=2)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def draw_panel_b(ax, standalone=False):
    """NISAR per-pair apparent vertical displacement vs ATL15 seasonal reference rates."""
    for (name, (mu, sd)), ls in zip(ref_rates.items(), ("-", (0, (4, 2)))):
        ax.axhspan(mu - sd, mu + sd, color=INK_3, alpha=0.13, lw=0, zorder=0.5)
        ax.axhline(mu, color=INK_2, lw=1.0, ls=ls, zorder=1)
        ax.text(np.datetime64("2026-06-17"), mu, f"{name}\n{CLIM_YEARS[0]}–{CLIM_YEARS[1]} mean ± SD", fontsize=6,
                color=INK_2, va="center", ha="left", path_effects=HALO, zorder=4)
    ax.axhline(0, color=INK_3, lw=0.5, zorder=0.5)
    first_valid = None
    for (t, d, ref, sec), r in sorted(pairs.items()):
        if r["cell_dv_m"] is None:
            continue
        t0, t1 = ymd(ref), ymd(sec)
        first_valid = t0 if first_valid is None else min(first_valid, t0)
        ax.plot([t0, t1], [r["cell_dv_m"]] * 2, color=C_NISAR, lw=2.4, solid_capstyle="butt", zorder=3)
        if r["pixel_dv_m"] is not None:
            ax.plot([t0, t1], [r["pixel_dv_m"]] * 2, color=C_NISAR, lw=0.9, ls=(0, (2, 1.5)), zorder=3)
        ax.plot(t0 + (t1 - t0) / 2, r["cell_dv_m"], marker=marker[(t, d)], color=C_NISAR, ms=6, mec="white",
                mew=0.7, ls="", zorder=4)
    ax.annotate(f"No valid NISAR pairs on the ice {np.datetime_as_string(nisar_start)} – "
                f"{np.datetime_as_string(first_valid)}:\nmelt-season decorrelation (coherence < {COHERENCE_MIN}"
                f"{', see coherence figure' if standalone else ', see c'})",
                (nisar_start + (first_valid - nisar_start) / 2, 0.93), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=6.5, color=INK_2, path_effects=HALO, zorder=5)
    ax.text(0.015, 0.04,  # lower left: no valid pairs there (melt season), so nothing is hidden
            "NISAR 'vertical' = LOS / cos θ (pipeline method), assuming purely vertical motion.\n"
            "On flowing ice the horizontal-flow component dominates LOS, so this is NOT surface-elevation\n"
            "change: ascending and descending tracks disagree in sign. Provisional data, referenced to\n"
            f"ice-free bedrock near Point 660;\n{CORRECTION_NOTE}.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e6", ec="#e0b85c", lw=0.7), zorder=6)
    ax.set_ylabel(f"Apparent vertical\nper pair (m)")
    ax.set_title(f"{'' if standalone else '(b) '}NISAR per-pair \"apparent vertical\" displacement vs ATL15 "
                 f"seasonal height change per {PAIR_DAYS} days", loc="left")
    if standalone:
        ax.set_xlim(np.datetime64("2026-06-15"), np.datetime64("2026-10-25"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        handles = (track_handles()
                   + [Line2D([], [], color=C_NISAR, lw=2.4, label="NISAR 1 km cell mean"),
                      Line2D([], [], color=C_NISAR, lw=0.9, ls=(0, (2, 1.5)), label="NISAR 80 m pixel at the crossover")]
                   + [Line2D([], [], color=INK_2, lw=1.0, ls=ls, label=f"{name}, {CLIM_YEARS[0]}–{CLIM_YEARS[1]} mean")
                      for name, ls in zip(ref_rates, ("-", (0, (4, 2))))]
                   + [Patch(facecolor=INK_3, alpha=0.13, label="ATL15 ±1 SD between years")])
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=6.6)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def draw_panel_c(ax, standalone=False):
    """Per-pair coherence: the 80 m pixel at the crossover (blue) vs the bedrock reference (gray)."""
    ax.axhline(COHERENCE_MIN, color=INK_2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(np.datetime64("2026-06-17"), COHERENCE_MIN + 0.015, f"pipeline threshold {COHERENCE_MIN}", fontsize=6,
            color=INK_2, va="bottom")
    for (t, d, ref, sec), r in sorted(pairs.items()):
        t0, t1 = ymd(ref), ymd(sec)
        mid = t0 + (t1 - t0) / 2
        ax.plot(mid, r["pixel_coh"], marker=marker[(t, d)], color=C_NISAR, ms=5.5, mec="white", mew=0.5, ls="",
                zorder=3)
        ax.plot(mid, r["ref_coh_median"], marker=marker[(t, d)], color=INK_3, ms=4, mec="white", mew=0.5, ls="",
                zorder=3)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Coherence")
    ax.set_xlim(np.datetime64("2026-06-15"), np.datetime64("2026-10-25"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title(f"{'' if standalone else '(c) '}NISAR coherence per pair: ice pixel at the crossover (blue) vs "
                 "bedrock reference (gray)", loc="left")
    if standalone:
        handles = (track_handles(5.5)
                   + [Line2D([], [], ls="", marker="o", color=C_NISAR, mec="white", ms=5,
                             label="Ice: 80 m pixel at the crossover"),
                      Line2D([], [], ls="", marker="o", color=INK_3, mec="white", ms=4,
                             label="Bedrock reference (median, ice-free)"),
                      Line2D([], [], color=INK_2, lw=0.8, ls=(0, (4, 2)),
                             label=f"Pipeline coherence threshold ({COHERENCE_MIN})")])
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6.6)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


# combined three-panel figure
fig = plt.figure(figsize=(7.5, 7.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 0.6], hspace=0.42)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])
axc = fig.add_subplot(gs[2], sharex=axb)
draw_panel_a(axa)
draw_panel_b(axb)
draw_panel_c(axc)
handles = (track_handles()
           + [Line2D([], [], color=C_NISAR, lw=2.4, label="NISAR 1 km cell mean"),
              Line2D([], [], color=C_NISAR, lw=0.9, ls=(0, (2, 1.5)), label="NISAR 80 m pixel at the crossover"),
              Line2D([], [], ls="", marker="o", color=INK_3, mec="white", ms=4, label="Bedrock reference coherence"),
              melt_patch])
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, -0.035, f"{SAMPLE_NOTE} {SOURCES_ATL15}\n{SOURCES_NISAR}", ha="center", va="top", fontsize=5.8,
         color=INK_2)
fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.14)
sc.save_figure(fig, FIGURE_OUTPUT)

# the same panels as separate figures, each with its own legend and source note
for draw, suffix, height, note in ((draw_panel_a, "a_ATL15", 3.0, f"{SAMPLE_NOTE}\n{SOURCES_ATL15}"),
                                   (draw_panel_b, "b_NISAR", 3.6, f"{SAMPLE_NOTE}\n{SOURCES_NISAR}"),
                                   (draw_panel_c, "c_coherence", 2.6, f"{SAMPLE_NOTE}\n{SOURCES_NISAR}")):
    fig, ax = plt.subplots(figsize=(7.5, height))
    draw(ax, standalone=True)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=5.8, color=INK_2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.3)
    sc.save_figure(fig, f"{FIGURE_OUTPUT}_{suffix}")
