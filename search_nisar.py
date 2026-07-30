# search_nisar.py
# Queries the NASA Earthdata Common Metadata Repository (CMR) for NISAR granules
# that intersect a user-supplied bounding box and fall within a temporal range.
# Outputs a summary of matching files — no data is downloaded.

import earthaccess  # NASA Earthdata authentication, search, and download library


def search_nisar_granules(
    short_name,   # NISAR CMR collection short name, e.g. 'NISAR_L2_GCOV'
    lat_min,      # southernmost latitude of the region of interest (decimal degrees)
    lat_max,      # northernmost latitude of the region of interest (decimal degrees)
    lon_min,      # westernmost longitude of the region of interest (decimal degrees)
    lon_max,      # easternmost longitude of the region of interest (decimal degrees)
    date_start,   # start of the temporal search window as 'YYYY-MM-DD'
    date_end,     # end of the temporal search window as 'YYYY-MM-DD'
    count=200,    # maximum number of granules to retrieve from CMR (default 200)
    orbit_direction=None,  # 'ascending', 'descending', or None to skip this filter
):
    """
    Search NASA Earthdata CMR for NISAR granules that spatially intersect a
    bounding box and whose acquisition date falls within [date_start, date_end].

    Returns a list of earthaccess DataGranule objects.  Each object exposes
    download URLs (.data_links()), file size (.size()), and full CMR metadata
    (.umm), and can be passed directly to earthaccess.download() in a later step.

    orbit_direction, if given, restricts results to ascending or descending
    passes (case-insensitive: 'ascending'/'descending'). NISAR granules encode
    this as the CMR additional attribute ASCENDING_DESCENDING, which isn't one
    of earthaccess's built-in search_data() keyword filters, so it's applied
    here as a raw CMR "attribute" query parameter instead.

    Authentication is attempted automatically from, in order:
      1. environment variables EARTHDATA_USERNAME / EARTHDATA_PASSWORD
      2. a ~/.netrc entry for urs.earthdata.nasa.gov
      3. an interactive username/password prompt
    """

    # -------------------------------------------------------------------------
    # Authentication
    # earthaccess.login() connects to NASA Earthdata (urs.earthdata.nasa.gov).
    # strategy="all" tries environment variables, then .netrc, then an
    # interactive prompt — whichever succeeds first is used.  The returned
    # Auth object is stored in the earthaccess module and used implicitly by
    # all subsequent search and download calls.
    # -------------------------------------------------------------------------
    auth = earthaccess.login(strategy="all")  # attempt login via all available credential sources
    print(f"Earthdata login status: authenticated={auth.authenticated}")  # confirm credentials were accepted

    # -------------------------------------------------------------------------
    # Spatial filter — bounding box
    # CMR's bounding_box argument expects the four corners in (W, S, E, N)
    # order, i.e. (lon_min, lat_min, lon_max, lat_max).  We construct that
    # tuple here from the individual coordinate parameters.
    # -------------------------------------------------------------------------
    bbox = (lon_min, lat_min, lon_max, lat_max)  # repack coordinates into CMR's (W, S, E, N) convention
    print(f"Spatial filter  — bounding box (W, S, E, N): {bbox}")  # log bbox so the user can verify orientation

    # -------------------------------------------------------------------------
    # Temporal filter
    # CMR accepts a two-element tuple of ISO-8601 date strings.  The range is
    # inclusive on both ends, so granules acquired exactly on date_start or
    # date_end are included.
    # -------------------------------------------------------------------------
    temporal = (date_start, date_end)  # pack start/end into a tuple for the CMR query
    print(f"Temporal filter — range: {temporal[0]}  →  {temporal[1]}")  # log the date window

    # -------------------------------------------------------------------------
    # Orbit direction filter (ascending / descending)
    # NISAR granules carry their pass direction as a CMR "additional attribute"
    # named ASCENDING_DESCENDING (values "ASCENDING" / "DESCENDING") rather than
    # as a dedicated top-level field, so earthaccess.search_data() has no
    # built-in keyword for it. CMR's additional-attribute filter is expressed
    # as attribute[]=<type>,<name>,<value>; python-cmr's GranuleQuery has no
    # wrapper method for that, but its underlying `.params` dict is exactly
    # what builds the query string, so we set it directly below.
    # -------------------------------------------------------------------------
    if orbit_direction is not None:
        orbit_direction = orbit_direction.strip().upper()
        if orbit_direction not in ("ASCENDING", "DESCENDING"):
            raise ValueError(
                f"orbit_direction must be 'ascending', 'descending', or None "
                f"(got {orbit_direction!r})"
            )
        print(f"Orbit direction filter: {orbit_direction}")  # log the requested pass direction

    # -------------------------------------------------------------------------
    # CMR granule search
    # Build the query the same way earthaccess.search_data() does internally
    # (DataGranules(auth).parameters(...).get(...)), so we can also attach the
    # additional-attribute orbit filter before executing it.
    #   short_name   — identifies the NISAR collection (product type + version)
    #   bounding_box — spatial filter; CMR returns granules whose footprint
    #                  intersects (not just falls within) the box
    #   temporal     — acquisition-date filter
    #   count        — upper bound on results; CMR pages internally so large
    #                  counts are fine but increase query time
    # The call is synchronous and returns when all matching pages are fetched.
    # -------------------------------------------------------------------------
    print(f"\nQuerying CMR for collection '{short_name}' …")  # status message before the network call
    query = earthaccess.DataGranules(auth).parameters(  # build the CMR query
        short_name=short_name,           # NISAR collection identifier (see NOTE below)
        bounding_box=bbox,               # spatial intersection filter
        temporal=temporal,               # acquisition-date range filter
    )
    if orbit_direction is not None:
        query.params["attribute"] = [f"string,ASCENDING_DESCENDING,{orbit_direction}"]  # raw CMR additional-attribute filter
    granules = query.get(count) if count > 0 else query.get_all()  # execute the CMR search and collect all result pages
    print(granules)
    # NOTE: NISAR collection short names follow the pattern NISAR_<level>_<product>,
    # e.g. NISAR_L2_GCOV, NISAR_L2_GSLC, NISAR_L2_GUNW, NISAR_L1_RSLC.
    # If CMR returns 0 results, verify the exact short name at:
    # https://search.earthdata.nasa.gov  (filter by "NISAR" provider)

    # -------------------------------------------------------------------------
    # Results summary
    # Print a one-line entry per granule showing its file name, acquisition
    # date, and size so the user can assess the result set before downloading.
    # -------------------------------------------------------------------------
    print(f"\nFound {len(granules)} granule(s) matching the query.\n")  # report total hit count
    for i, granule in enumerate(granules):                               # iterate over each returned granule
        urls = granule.data_links()                                      # list of download URL strings for this granule
        filename = urls[0].split("/")[-1] if urls else "(no URL)"        # last path segment of the first URL = filename
        size_mb = granule.size()                                         # granule file size in megabytes (float)
        print(f"  [{i + 1:>4}]  {filename:<60}  {size_mb:>8.1f} MB")    # formatted table row: index, filename, size

    # -------------------------------------------------------------------------
    # Return
    # Return the raw granule list so this function composes naturally with
    # downstream steps such as earthaccess.download(granules, local_path=...).
    # -------------------------------------------------------------------------
    return granules  # list of earthaccess DataGranule objects


# =============================================================================
# Entry point / test parameters
# This block runs when the file is executed directly (python search_nisar.py)
# or via F5 / "Run file" in Spyder.  Edit the values here to test different
# products, regions, or date windows without touching the function above.
# After a run, the `results` variable is available in the Spyder variable
# explorer for interactive inspection.
# =============================================================================
if __name__ == "__main__":

    # --- test parameters -------------------------------------------------------
    #TEST_SHORT_NAME = "NISAR_L2_GUNW_BETA_V1"  # NISAR Level-2 Geocoded Unwrapped Interferogram (Beta V1)
    TEST_SHORT_NAME = "NISAR_L2_GUNW_PROVISIONAL_V1"
    TEST_LAT_MIN    = 68.6515            # southern boundary — Brooks Range, Alaska
    TEST_LAT_MAX    = 68.6542              # northern boundary — Brooks Range, Alaska
    TEST_LON_MIN    = -149.6247             # western boundary — Brooks Range, Alaska
    TEST_LON_MAX    = -149.5438            # eastern boundary — Brooks Range, Alaska
    TEST_DATE_START = "2026-06-01"           # start of search window
    TEST_DATE_END   = "2026-08-01"           # end of search window
    TEST_COUNT      = 100                     # cap at 50 results for a quick test run
    TEST_ORBIT_DIRECTION = 'ascending'              # 'ascending', 'descending', or None for both

    # --- execute search --------------------------------------------------------
    results = search_nisar_granules(    # call the search function with test parameters
        short_name  = TEST_SHORT_NAME,  # NISAR collection to query
        lat_min     = TEST_LAT_MIN,     # southern edge of the region of interest
        lat_max     = TEST_LAT_MAX,     # northern edge of the region of interest
        lon_min     = TEST_LON_MIN,     # western edge of the region of interest
        lon_max     = TEST_LON_MAX,     # eastern edge of the region of interest
        date_start  = TEST_DATE_START,  # beginning of the temporal window
        date_end    = TEST_DATE_END,    # end of the temporal window
        count       = TEST_COUNT,       # maximum granules to return
        orbit_direction = TEST_ORBIT_DIRECTION,  # 'ascending', 'descending', or None
    )
    # `results` is now a list of DataGranule objects; inspect in Spyder or
    # print individual metadata with: results[0].umm  (full CMR UMM-G record)
