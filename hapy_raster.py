"""
Standalone rasterization function for unstructured grid data.

This module provides a to_raster function that works directly with xarray
DataArrays/Datasets without requiring the uxarray library. It's designed
for plotting E3SM/EAMxx data on unstructured grids using SCRIP format grid files.

Nearest-neighbor lookups are done with a KDTree built on 3D unit-sphere
coordinates, so great-circle nearest neighbors are found exactly everywhere
(including the poles) without any special-case polar handling.

Usage:
    import xarray as xr
    from to_raster import to_raster

    # Load data and grid
    data = xr.open_dataset('data.nc')['variable']
    grid = xr.open_dataset('grid_scrip.nc')

    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    raster = to_raster(data, grid, ax=ax)
    ax.imshow(raster, origin='lower', extent=ax.get_xlim() + ax.get_ylim())
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import cartopy.crs as ccrs

# maximum number of cached pixel->column mappings (see _QUERY_CACHE below)
_CACHE_MAX_ENTRIES = 4


def _haversine_distance_matrix(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distances using the Haversine formula.

    This properly handles the spherical geometry of Earth, including
    longitude wrapping and polar regions.

    Parameters
    ----------
    lon1, lat1 : array_like
        Coordinates of first points (in degrees)
    lon2, lat2 : array_like
        Coordinates of second points (in degrees)

    Returns
    -------
    distances : ndarray
        Great circle distances in degrees
    """
    # Convert to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Return in degrees
    return np.degrees(c)


def _lonlat_to_xyz(lon, lat):
    """
    Convert lon/lat (degrees) to Cartesian coordinates on the unit sphere.

    Returns an (N,3) float64 array suitable for building a cKDTree.
    """
    lon_rad = np.radians(np.asarray(lon, dtype=np.float64))
    lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    xyz = np.empty((lon_rad.size, 3), dtype=np.float64)
    xyz[:, 0] = cos_lat * np.cos(lon_rad)
    xyz[:, 1] = cos_lat * np.sin(lon_rad)
    xyz[:, 2] = np.sin(lat_rad)
    return xyz


def _chord_to_degrees(chord):
    """
    Convert unit-sphere chord length to great-circle distance in degrees.

    Chord length is a monotonic function of great-circle distance, so nearest
    neighbors found with chord distance are also the great-circle nearest
    neighbors - this conversion is only needed to report distances in degrees.
    """
    return np.degrees(2.0 * np.arcsin(np.clip(np.asarray(chord)/2.0, 0.0, 1.0)))


class SphericalKDTree:
    """
    KDTree for great-circle nearest-neighbor lookups on the sphere.

    Points are mapped to 3D Cartesian coordinates on the unit sphere before
    the tree is built, so nearest neighbors are exact at all latitudes and
    longitude wrapping is handled implicitly. Query cost is O(log N) per
    point rather than O(N).
    """

    def __init__(self, lon, lat, polar_threshold=None):
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        # retained for backward compatibility - no longer used
        self.polar_threshold = polar_threshold

        self.xyz = _lonlat_to_xyz(self.lon, self.lat)
        # sliding-midpoint build is much cheaper than a balanced build for the
        # tens of millions of points in a high-res unstructured grid
        self.tree = cKDTree(self.xyz, compact_nodes=False, balanced_tree=False)

    def query(self, query_lon, query_lat, k=1, workers=-1):
        """
        Query the tree for the k nearest neighbors of each (lon,lat) point.

        Returns (distances, indices) with distances in degrees of great-circle arc.
        """
        query_xyz = _lonlat_to_xyz(np.atleast_1d(query_lon), np.atleast_1d(query_lat))
        chord, indices = self.tree.query(query_xyz, k=k, workers=workers)
        return _chord_to_degrees(chord), indices

    def typical_spacing(self, sample_size=1000):
        """
        Nearest-neighbor separations (degrees) for a random sample of grid points.
        """
        n = self.xyz.shape[0]
        if n < 2: return None
        sample_size = min(sample_size, n)
        # fixed seed so repeated runs give identical rasters, and integers()
        # rather than choice(replace=False) - the latter permutes all N points
        sample = np.random.default_rng(0).integers(0, n, sample_size)
        chord, _ = self.tree.query(self.xyz[sample], k=2, workers=-1)
        return _chord_to_degrees(chord[:, 1])


#---------------------------------------------------------------------------------------------------
# Cache of pixel->column mappings. Rasterizing several panels that share the same
# grid and the same axes geometry repeats an identical tree build and pixel query,
# which dominates the cost for large grids, so the mapping is reused.
_QUERY_CACHE = {}


def clear_raster_cache():
    """Drop all cached trees and pixel->column mappings."""
    _QUERY_CACHE.clear()


def _fingerprint(arr):
    """Cheap O(1) identity fingerprint for a coordinate array."""
    n = arr.size
    if n == 0: return (0,)
    def _s(v):
        v = float(v)
        return v if np.isfinite(v) else 'nonfinite'
    return (n, str(arr.dtype), _s(arr[0]), _s(arr[n//2]), _s(arr[-1]))


def _cache_store(key, entry):
    if len(_QUERY_CACHE) >= _CACHE_MAX_ENTRIES:
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[key] = entry


def _build_mapping(lon, lat, keep, lon_query, lat_query, valid_query, method, verbose):
    """
    Build the pixel->column mapping for one grid/axes combination.

    Returns a dict with either:
        'nearest': src_index (n_pixel,) int64, -1 where the pixel has no data
        'linear' : src_index (n_pixel,k) int64 and weights (n_pixel,k), plus a
                   'valid' mask marking pixels with data
    Indices refer to positions in the original (unfiltered) column dimension.
    """
    keep_pos = np.flatnonzero(keep) if keep is not None else None
    if keep_pos is None:
        lon_clean, lat_clean = lon, lat
    else:
        lon_clean, lat_clean = lon[keep_pos], lat[keep_pos]

    if len(lon_clean) == 0:
        raise ValueError("No valid grid points found after removing NaNs")

    if verbose: print(f'Building KDTree over {len(lon_clean)} grid points...')
    tree = SphericalKDTree(lon_clean, lat_clean)

    # pixels farther than this from any column are treated as "no coverage".
    # Use a high percentile rather than the median so that variable-resolution
    # grids are not blanked out over their coarse regions.
    spacing = tree.typical_spacing()
    max_dist = np.percentile(spacing, 99) * 3 if spacing is not None else np.inf
    if verbose:
        print(f'done. grid spacing: median={np.median(spacing):.4g} '
              f'p99={np.percentile(spacing,99):.4g} deg -> max_dist={max_dist:.4g} deg')

    k = 1 if method == 'nearest' else min(4, len(lon_clean))
    if verbose: print(f'Querying {int(np.sum(valid_query))} pixels...')
    distances, indices = tree.query(lon_query[valid_query], lat_query[valid_query], k=k)
    if verbose: print('done.')

    # map back to positions in the original column dimension
    if keep_pos is not None: indices = keep_pos[indices]

    n_pixel = valid_query.size
    if method == 'nearest' or k == 1:
        if indices.ndim > 1: indices, distances = indices[:, 0], distances[:, 0]
        src_index = np.full(n_pixel, -1, dtype=np.int64)
        src_index[valid_query] = np.where(distances > max_dist, -1, indices)
        return {'method': 'nearest', 'src_index': src_index}

    weights = 1.0 / (distances + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    valid = valid_query.copy()
    valid[valid_query] = distances[:, 0] <= max_dist
    src_index = np.zeros((n_pixel, k), dtype=np.int64)
    wgt = np.zeros((n_pixel, k), dtype=np.float64)
    src_index[valid_query] = indices
    wgt[valid_query] = weights
    return {'method': 'linear', 'src_index': src_index, 'weights': wgt, 'valid': valid}


def _apply_mapping(mapping, values):
    """Gather data values through a pixel->column mapping, filling NaN elsewhere."""
    if mapping['method'] == 'nearest':
        src_index = mapping['src_index']
        raster_values = np.full(src_index.size, np.nan)
        ok = src_index >= 0
        raster_values[ok] = values[src_index[ok]]
        return raster_values
    valid = mapping['valid']
    raster_values = np.full(valid.size, np.nan)
    gathered = values[mapping['src_index'][valid]]
    raster_values[valid] = (gathered * mapping['weights'][valid]).sum(axis=1)
    return raster_values


def to_raster(data, grid, ax, pixel_ratio=1.0, method='nearest',
              use_spherical=True, polar_threshold=60.0, verbose=False):
    """
    Convert unstructured grid data to a raster array for plotting.

    This function takes data on an unstructured grid (with 'ncol' dimension)
    and converts it to a regular 2D raster array suitable for display with
    matplotlib's imshow, contour, or contourf functions. It samples the
    unstructured grid at each pixel location in the provided GeoAxes.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data to be rasterized. Must have an 'ncol' dimension and associated
        'lat' and 'lon' coordinates. For Dataset input, only the first data
        variable will be used.

        Expected structure:
            <xarray.DataArray 'variable' (ncol: N)>
            Coordinates:
                lat  (ncol) float64
                lon  (ncol) float64

    grid : xarray.Dataset
        Grid information from a SCRIP format file. Must contain:
            - grid_center_lat (grid_size) float64
            - grid_center_lon (grid_size) float64

        Expected structure:
            <xarray.Dataset>
            Dimensions: (grid_size: N)
            Data variables:
                grid_center_lat  (grid_size) float64
                grid_center_lon  (grid_size) float64

    ax : cartopy.mpl.geoaxes.GeoAxes
        Cartopy GeoAxes object with a projection. The raster resolution is
        determined from the axes bounds and pixel_ratio.

    pixel_ratio : float, optional
        Multiplier for the raster resolution. Default is 1.0.
        - pixel_ratio < 1.0: Lower resolution (faster, coarser)
        - pixel_ratio = 1.0: Default resolution (1 raster pixel per screen pixel)
        - pixel_ratio > 1.0: Higher resolution (slower, finer)

    method : str, optional
        Interpolation method. Options:
        - 'nearest': Nearest neighbor (fastest, default)
        - 'linear': Inverse distance weighting over nearest neighbors (slower)
        Default is 'nearest'.

    use_spherical, polar_threshold : optional
        Deprecated and ignored. Lookups now always use exact great-circle
        nearest neighbors via a 3D unit-sphere KDTree, which is both faster
        and more accurate than either of the paths these used to select.

    Returns
    -------
    numpy.ndarray
        2D array of rasterized values with shape (height, width) determined
        by the axes bounds and pixel_ratio. Values correspond to the data
        sampled at each pixel location. NaN values indicate pixels where
        no valid data was found.

    Notes
    -----
    - The pixel->column mapping is cached, so repeated calls that share the same
      grid and the same axes geometry (e.g. multiple panels of a figure) skip
      the tree build and the pixel query entirely. Use clear_raster_cache() to
      release that memory.
    - For Dask arrays, data is computed before rasterization
    - Longitude values are normalized to [-180, 180] for consistency
    - Properly handles different Cartopy projections (Orthographic, Robinson, etc.)
    - The returned array is suitable for use with:
        ax.imshow(raster, origin='lower', extent=ax.get_xlim() + ax.get_ylim())

    Examples
    --------
    >>> import xarray as xr
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>>
    >>> # Load data and grid
    >>> data = xr.open_dataset('data.nc')['TREFHT'].isel(time=0)
    >>> grid = xr.open_dataset('ne30pg3_scrip.nc')
    >>>
    >>> # Create raster plot with Orthographic projection
    >>> fig, ax = plt.subplots(
    ...     subplot_kw={'projection': ccrs.Orthographic(central_latitude=-85)},
    ...     figsize=(12, 8)
    ... )
    >>> ax.set_global()
    >>> raster = to_raster(data, grid, ax=ax, pixel_ratio=1.0)
    >>> img = ax.imshow(
    ...     raster,
    ...     cmap='RdBu_r',
    ...     origin='lower',
    ...     extent=ax.get_xlim() + ax.get_ylim()
    ... )
    >>> ax.coastlines()
    >>> plt.colorbar(img, ax=ax)
    >>> plt.show()
    """

    if method not in ['nearest','linear']:
        raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'linear'")

    # Handle Dataset input - extract first data variable
    if isinstance(data, xr.Dataset):
        data_vars = list(data.data_vars.keys())
        if not data_vars:
            raise ValueError("Dataset contains no data variables")
        data = data[data_vars[0]]
        if verbose: print(f"Note: Using data variable '{data_vars[0]}' from Dataset")

    # Verify data has required structure
    if 'ncol' not in data.dims:
        raise ValueError(f"Data must have 'ncol' dimension. Found dimensions: {list(data.dims)}")

    # Get coordinates from data (preferred) or grid
    if 'lat' in data.coords and 'lon' in data.coords:
        lat = data.coords['lat'].values
        lon = data.coords['lon'].values
    elif 'grid_center_lat' in grid and 'grid_center_lon' in grid:
        lat = grid['grid_center_lat'].values
        lon = grid['grid_center_lon'].values
    else:
        raise ValueError(
            "Neither data coordinates (lat/lon) nor grid coordinates "
            "(grid_center_lat/grid_center_lon) found"
        )

    # Compute data if it's a Dask array
    if hasattr(data, 'chunks') and data.chunks is not None:
        if verbose: print("Computing Dask array...")
        values = data.compute().values
        if verbose: print('done.')
    else:
        values = data.values

    # Normalize longitude to [-180, 180]
    lon = np.where(lon > 180, lon - 360, lon)

    # Get the projection from the axes
    projection = ax.projection

    # Get axes bounds in projection coordinates
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Determine raster resolution from axes
    bbox = ax.get_window_extent()
    width_px = int(bbox.width * pixel_ratio)
    height_px = int(bbox.height * pixel_ratio)

    # Ensure minimum resolution
    width_px = max(width_px, 10)
    height_px = max(height_px, 10)

    #-------------------------------------------------------------------------
    # Points where the data itself is NaN are excluded from the tree so that
    # nearby valid data fills them in. That depends on the data values, so it
    # cannot use the cache shared across panels.
    coord_valid = ~(np.isnan(lon) | np.isnan(lat))
    data_has_nan = bool(np.isnan(values).any())
    if data_has_nan:
        keep = coord_valid & ~np.isnan(values)
    else:
        keep = None if coord_valid.all() else coord_valid

    cache_key = None
    if not data_has_nan:
        cache_key = ( _fingerprint(lon), _fingerprint(lat), repr(projection),
                      x_min, x_max, y_min, y_max, width_px, height_px, method )
        mapping = _QUERY_CACHE.get(cache_key)
        if mapping is not None:
            if verbose: print('Using cached pixel mapping')
            return _apply_mapping(mapping, values).reshape(height_px, width_px)

    #-------------------------------------------------------------------------
    # Create raster grid in PROJECTION coordinates
    x_range = np.linspace(x_min, x_max, width_px)
    y_range = np.linspace(y_min, y_max, height_px)
    x_proj, y_proj = np.meshgrid(x_range, y_range)

    # Transform projection coordinates back to lon/lat for lookup
    # This is the key step for handling different projections
    geo_crs = ccrs.PlateCarree()

    x_proj_flat = x_proj.ravel()
    y_proj_flat = y_proj.ravel()

    try:
        points_geo = geo_crs.transform_points(projection, x_proj_flat, y_proj_flat)
        lon_query = points_geo[:, 0]
        lat_query = points_geo[:, 1]
    except Exception as e:
        if verbose: print(f"Warning: Transform failed with error: {e}")
        if verbose: print("Falling back to direct coordinate usage")
        lon_query = x_proj_flat
        lat_query = y_proj_flat

    # Handle points that fall outside the valid projection domain
    # (e.g., back of globe in Orthographic projection)
    valid_query = ~(np.isnan(lon_query) | np.isnan(lat_query) |
                    np.isinf(lon_query) | np.isinf(lat_query))

    # Normalize query longitudes to [-180, 180]
    lon_query = np.where(lon_query > 180, lon_query - 360, lon_query)
    lon_query = np.where(lon_query < -180, lon_query + 360, lon_query)

    #-------------------------------------------------------------------------
    mapping = _build_mapping(lon, lat, keep, lon_query, lat_query,
                             valid_query, method, verbose)
    if cache_key is not None: _cache_store(cache_key, mapping)

    return _apply_mapping(mapping, values).reshape(height_px, width_px)


def to_raster_with_mask(data, grid, ax, pixel_ratio=1.0, method='nearest',
                        mask_value=None, use_spherical=True, polar_threshold=60.0):
    """
    Extended version of to_raster with support for masked values.

    This is useful when you want to mask specific values (e.g., ocean/land mask)
    before rasterization.

    Parameters
    ----------
    data : xarray.DataArray
        Data to be rasterized
    grid : xarray.Dataset
        Grid information from SCRIP file
    ax : cartopy.mpl.geoaxes.GeoAxes
        Cartopy GeoAxes object
    pixel_ratio : float, optional
        Raster resolution multiplier
    method : str, optional
        Interpolation method ('nearest' or 'linear')
    mask_value : float or None, optional
        Value to mask in the data before rasterization. These values will
        be treated as NaN.
    use_spherical, polar_threshold : optional
        Deprecated and ignored, see to_raster.

    Returns
    -------
    numpy.ndarray
        2D rasterized array
    """
    if mask_value is not None:
        data = data.where(data != mask_value)

    return to_raster(data, grid, ax, pixel_ratio=pixel_ratio, method=method)
