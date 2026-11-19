# icesat2_tracks.py
# Helpers for working with ICESat-2 nominal Reference Ground Tracks (RGTs):
# load the official per-cycle RGT KML archive for an area of interest,
# project passes to a map CRS, build the six beam ground tracks around each
# RGT, find ascending x descending crossovers, and list nominal pass dates
# under the 91-day exact-repeat orbit. Importing this module has no side
# effects; run it directly for a self-check against the Toolik site.

import re                              # parse KML coordinate/timestamp text
import zipfile                         # read the per-RGT KMLs straight out of the archive
from dataclasses import dataclass      # lightweight per-pass container
from datetime import datetime, timedelta
import numpy as np                     # array operations
from pyproj import Transformer         # lon/lat -> projected map coordinates


# =============================================================================
# Parameters / nominal ICESat-2 geometry
# RGT file: "IS2_RGTs_cycle<N>_date_time.zip" from the ICESat-2 technical
# specs page (https://icesat-2.gsfc.nasa.gov/science/specs). Each KML holds a
# LineString with one vertex per second of flight time, plus one timestamped
# point per minute whose position coincides with every 60th vertex — so a
# vertex's time is simply the first timestamp + its index in seconds.
# Beam layout: 3 beam pairs ~3.3 km apart cross-track, with the RGT along the
# centre pair; beams within a pair are ~90 m apart (strong/weak). ATL06 land
# ice heights are 40 m segments posted every 20 m along each beam.
# (Neumann et al. 2019, RSE, doi:10.1016/j.rse.2019.111325; Smith et al.
# 2019, RSE, doi:10.1016/j.rse.2019.111352)
# =============================================================================
RGT_ZIP_PATH      = "./data/icesat2_rgt/IS2_RGTs_cycle32_date_time.zip"  # nominal RGTs, cycle 32 (Jun-Sep 2026)
REPEAT_DAYS       = 91.0      # exact-repeat period: 1387 orbits per 91 days
PAIR_SPACING_M    = 3300.0    # cross-track spacing between beam pairs (meters)
BEAM_SEPARATION_M = 90.0      # cross-track spacing between the two beams of a pair (meters)
ATL06_POSTING_M   = 20.0      # ATL06 along-track segment-centre spacing (meters)
ATL06_SEGMENT_M   = 40.0      # ATL06 segment length (meters)
FOOTPRINT_M       = 11.0      # approximate laser footprint diameter (meters)

_TIME_RE   = re.compile(r"<description>RGT\s+\d+\s*\n\s*(\d{2}-\w{3}-\d{4} \d{2}:\d{2}:\d{2})")
_LINE_RE   = re.compile(r"<LineString.*?<coordinates>(.*?)</coordinates>", re.S)
_NAME_RE   = re.compile(r"IS2_RGT_(\d{4})_")


@dataclass
class Pass:
    """One contiguous pass of an RGT through the area of interest."""
    rgt: int              # reference ground track number (1-1387)
    lon: np.ndarray       # vertex longitudes (degrees)
    lat: np.ndarray       # vertex latitudes (degrees)
    t: np.ndarray         # vertex UTC times (numpy datetime64[ms])
    ascending: bool       # True if latitude increases along the pass
    x: np.ndarray = None  # projected easting (meters), filled by project_passes()
    y: np.ndarray = None  # projected northing (meters), filled by project_passes()


def load_rgts(lon_min, lat_min, lon_max, lat_max, zip_path=RGT_ZIP_PATH, margin_deg=0.5):
    """
    Read every RGT in the archive and return the passes that cross the box
    (expanded by margin_deg so lines reach past the map edges). Each pass is
    one contiguous run of vertices inside the expanded box.
    """
    box = (lon_min - margin_deg / np.cos(np.radians(lat_max)), lat_min - margin_deg,
           lon_max + margin_deg / np.cos(np.radians(lat_max)), lat_max + margin_deg)
    passes = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".kml"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            coords = np.array(_LINE_RE.search(text).group(1).replace(",", " ").split(), float).reshape(-1, 3)
            lon, lat = coords[:, 0], coords[:, 1]
            inside = (lon >= box[0]) & (lon <= box[2]) & (lat >= box[1]) & (lat <= box[3])
            if not inside.any():
                continue
            t0 = np.datetime64(datetime.strptime(_TIME_RE.search(text).group(1), "%d-%b-%Y %H:%M:%S"), "ms")
            t = t0 + np.arange(len(lon)) * np.timedelta64(1000, "ms")  # one vertex per second
            rgt = int(_NAME_RE.search(name).group(1))
            # split into contiguous runs, padding one vertex each side so the line crosses the box edge
            idx = np.flatnonzero(inside)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                sl = slice(max(run[0] - 1, 0), min(run[-1] + 2, len(lon)))
                passes.append(Pass(rgt, lon[sl], lat[sl], t[sl], bool(lat[sl][-1] > lat[sl][0])))
    return passes


def project_passes(passes, crs):
    """Fill each pass's x/y in the given projected CRS (an EPSG code or any pyproj CRS string)."""
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{crs}" if isinstance(crs, int) else crs, always_xy=True)
    for p in passes:
        p.x, p.y = to_map.transform(p.lon, p.lat)
        p.x, p.y = np.asarray(p.x), np.asarray(p.y)
    return passes


def _unit_normals(x, y):
    """Per-vertex unit normal pointing to the left of the direction of travel."""
    tx, ty = np.gradient(x), np.gradient(y)
    norm = np.hypot(tx, ty)
    return -ty / norm, tx / norm


def beam_tracks(p, include_weak=True):
    """
    Offset the RGT centreline into beam ground tracks. Returns a dict mapping
    cross-track offset (meters, positive = left of travel) to (x, y) arrays.
    Over a few tens of km the 1-s vertex spacing (~7 km) is short enough that
    straight segments in the map projection deviate from the true track by ~1 m.
    """
    nx, ny = _unit_normals(p.x, p.y)
    half = BEAM_SEPARATION_M / 2
    offsets = []
    for centre in (-PAIR_SPACING_M, 0.0, PAIR_SPACING_M):
        offsets += [centre - half, centre + half] if include_weak else [centre]
    return {off: (p.x + off * nx, p.y + off * ny) for off in offsets}


def resample(x, y, t, spacing_m):
    """Linearly resample a polyline (and its times) at a fixed along-track spacing."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])  # along-track distance (m)
    s_new = np.arange(0.0, s[-1], spacing_m)
    t_ms = t.astype("datetime64[ms]").astype(np.int64)
    return (np.interp(s_new, s, x), np.interp(s_new, s, y),
            np.interp(s_new, s, t_ms).astype("datetime64[ms]"))


def nearest_on_pass(p, x0, y0):
    """
    Closest point on a pass's centreline to (x0, y0). Returns a dict with the
    foot point, its UTC time, the distance, and the signed cross-track offset
    (positive = left of travel).
    """
    ax, ay, bx, by = p.x[:-1], p.y[:-1], p.x[1:], p.y[1:]
    dx, dy = bx - ax, by - ay
    f = np.clip(((x0 - ax) * dx + (y0 - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)  # fraction along each segment
    fx, fy = ax + f * dx, ay + f * dy
    d = np.hypot(x0 - fx, y0 - fy)
    i = int(np.argmin(d))
    t_ms = p.t.astype(np.int64)
    t_foot = np.datetime64(int(t_ms[i] + f[i] * (t_ms[i + 1] - t_ms[i])), "ms")
    side = np.sign(dx[i] * (y0 - ay[i]) - dy[i] * (x0 - ax[i]))  # cross product: + if point is left of travel
    return {"x": fx[i], "y": fy[i], "t": t_foot, "distance_m": d[i], "offset_m": side * d[i]}


def find_crossovers(passes):
    """
    All ascending x descending centreline intersections, in projected
    coordinates. Returns a list of dicts: x, y, the two RGT numbers, and the
    two nominal pass times at the crossover.
    """
    asc = [p for p in passes if p.ascending]
    desc = [p for p in passes if not p.ascending]
    out = []
    for pa in asc:
        for pd in desc:
            # vectorised segment-segment test over every (segment of pa) x (segment of pd)
            p0 = np.stack([pa.x[:-1], pa.y[:-1]], -1)[:, None, :]
            r = np.stack([np.diff(pa.x), np.diff(pa.y)], -1)[:, None, :]
            q0 = np.stack([pd.x[:-1], pd.y[:-1]], -1)[None, :, :]
            s = np.stack([np.diff(pd.x), np.diff(pd.y)], -1)[None, :, :]
            rxs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
            qp = q0 - p0
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / rxs  # fraction along pa's segment
                v = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / rxs  # fraction along pd's segment
            hit = (rxs != 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
            for i, j in zip(*np.nonzero(hit)):
                ta, td = pa.t.astype(np.int64), pd.t.astype(np.int64)
                out.append({
                    "x": pa.x[i] + u[i, j] * r[i, 0, 0],
                    "y": pa.y[i] + u[i, j] * r[i, 0, 1],
                    "rgt_asc": pa.rgt, "rgt_desc": pd.rgt,
                    "t_asc": np.datetime64(int(ta[i] + u[i, j] * (ta[i + 1] - ta[i])), "ms"),
                    "t_desc": np.datetime64(int(td[j] + v[i, j] * (td[j + 1] - td[j])), "ms"),
                })
    return out


def pass_dates(t_pass, start, end):
    """Nominal pass times within [start, end], stepping t_pass by whole 91-day repeat cycles."""
    t_pass = np.datetime64(t_pass, "ms")
    start, end = np.datetime64(start, "ms"), np.datetime64(end, "ms")
    step = np.timedelta64(int(REPEAT_DAYS * 86400e3), "ms")
    k0 = int(np.floor((start - t_pass) / step))
    k1 = int(np.ceil((end - t_pass) / step))
    times = t_pass + np.arange(k0, k1 + 1) * step
    return times[(times >= start) & (times <= end)]


# =============================================================================
# Self-check: load the Toolik canvas area and print basic geometry facts.
# Real ATL06 granules over Toolik (CMR, 2025-2026) are on RGTs 0213 and 0327,
# so those should appear among the nearest tracks.
# =============================================================================
if __name__ == "__main__":
    LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -150.3, 68.4, -148.9, 68.9  # around the Toolik canvas
    X0_LON, X0_LAT = -149.5869, 68.6521                               # Toolik sample point
    EPSG = 3413

    passes = project_passes(load_rgts(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), EPSG)
    print(f"{len(passes)} passes "
          f"({sum(p.ascending for p in passes)} ascending, {sum(not p.ascending for p in passes)} descending)")
    dt = np.diff(passes[0].t).astype(np.int64)
    print(f"Vertex time step: {dt.min()}-{dt.max()} ms")

    x0, y0 = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(X0_LON, X0_LAT)
    near = sorted(((nearest_on_pass(p, x0, y0), p) for p in passes), key=lambda a: a[0]["distance_m"])
    for n, p in near[:4]:
        print(f"  RGT {p.rgt:04d} {'asc ' if p.ascending else 'desc'}: {n['distance_m'] / 1000:6.2f} km "
              f"from sample point, pass {n['t']}")

    # adjacent same-direction track spacing near the sample point
    for asc in (True, False):
        d = sorted(n["distance_m"] * np.sign(n["offset_m"]) for n, p in near if p.ascending == asc)
        print(f"  {'Ascending ' if asc else 'Descending'} centreline spacing near point (km): "
              f"{np.round(np.diff(d) / 1000, 2)}")

    xo = find_crossovers(passes)
    print(f"{len(xo)} crossovers in area")
    c = min(xo, key=lambda c: np.hypot(c["x"] - x0, c["y"] - y0))
    for p in passes:  # re-measure the nearest crossover's distance to both of its lines
        if p.rgt in (c["rgt_asc"], c["rgt_desc"]):
            print(f"  nearest crossover RGT {c['rgt_asc']:04d} x {c['rgt_desc']:04d}: "
                  f"{np.hypot(c['x'] - x0, c['y'] - y0) / 1000:.2f} km away; "
                  f"distance to RGT {p.rgt:04d} line = {nearest_on_pass(p, c['x'], c['y'])['distance_m']:.3f} m")
    print("Nominal 2026 passes, asc RGT at crossover:",
          [str(t)[:16] for t in pass_dates(c["t_asc"], "2026-01-01", "2026-12-31")])
