"""
Standalone rasterization function for unstructured grid data.

This module provides a to_raster function that works directly with xarray
DataArrays/Datasets without requiring the uxarray library. It's designed
for plotting E3SM/EAMxx data on unstructured grids using SCRIP format grid files.

Usage:
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
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


def to_raster(data, grid, ax, pixel_ratio=1.0, method='nearest'):
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
        - 'linear': Linear interpolation using nearest neighbors (slower)
        Default is 'nearest'.
    
    Returns
    -------
    numpy.ndarray
        2D array of rasterized values with shape (height, width) determined
        by the axes bounds and pixel_ratio. Values correspond to the data
        sampled at each pixel location. NaN values indicate pixels where
        no valid data was found.
    
    Notes
    -----
    - The function uses a KDTree for efficient nearest-neighbor lookups
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
    
    # Handle Dataset input - extract first data variable
    if isinstance(data, xr.Dataset):
        data_vars = list(data.data_vars.keys())
        if not data_vars:
            raise ValueError("Dataset contains no data variables")
        data = data[data_vars[0]]
        print(f"Note: Using data variable '{data_vars[0]}' from Dataset")
    
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
        print("hapy.to_raster - Computing Dask array...")
        values = data.compute().values
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
    
    # Create raster grid in PROJECTION coordinates
    x_range = np.linspace(x_min, x_max, width_px)
    y_range = np.linspace(y_min, y_max, height_px)
    x_proj, y_proj = np.meshgrid(x_range, y_range)
    
    # Transform projection coordinates back to lon/lat for lookup
    # This is the key step for handling different projections
    geo_crs = ccrs.PlateCarree()
    
    # Flatten for transformation
    x_proj_flat = x_proj.ravel()
    y_proj_flat = y_proj.ravel()
    
    # Transform from projection coordinates to geographic coordinates (lon/lat)
    # Use transform_points which handles the conversion
    points_proj = np.column_stack([x_proj_flat, y_proj_flat])
    
    # Transform projection coordinates to geographic (lon/lat)
    try:
        points_geo = geo_crs.transform_points(
            projection, 
            x_proj_flat, 
            y_proj_flat
        )
        lon_query = points_geo[:, 0]
        lat_query = points_geo[:, 1]
    except Exception as e:
        print(f"Warning: Transform failed with error: {e}")
        print("Falling back to direct coordinate usage")
        lon_query = x_proj_flat
        lat_query = y_proj_flat
    
    # Handle points that fall outside the valid projection domain
    # (e.g., back of globe in Orthographic projection)
    valid_query = ~(np.isnan(lon_query) | np.isnan(lat_query) | 
                    np.isinf(lon_query) | np.isinf(lat_query))
    
    # Normalize query longitudes to [-180, 180]
    lon_query = np.where(lon_query > 180, lon_query - 360, lon_query)
    lon_query = np.where(lon_query < -180, lon_query + 360, lon_query)
    
    # Build KDTree from unstructured grid coordinates
    grid_points = np.column_stack([lon, lat])
    
    # Handle potential NaN values in grid
    valid_grid = ~(np.isnan(grid_points).any(axis=1) | np.isnan(values))
    grid_points = grid_points[valid_grid]
    values_clean = values[valid_grid]
    
    if len(grid_points) == 0:
        raise ValueError("No valid grid points found after removing NaNs")
    
    # Build KDTree for fast nearest-neighbor lookup
    tree = cKDTree(grid_points)
    
    # Initialize output with NaN
    raster_values = np.full(len(lon_query), np.nan)
    
    # Only query valid points
    if np.any(valid_query):
        query_points = np.column_stack([lon_query[valid_query], lat_query[valid_query]])
        
        if method == 'nearest':
            # Simple nearest neighbor
            distances, indices = tree.query(query_points, k=1)
            raster_values[valid_query] = values_clean[indices]
            
            # Set pixels too far from any grid point to NaN
            if len(grid_points) > 1:
                # Estimate grid spacing from first few points
                sample_size = min(100, len(grid_points))
                sample_distances, _ = tree.query(grid_points[:sample_size], k=2)
                typical_spacing = np.median(sample_distances[:, 1]) * 3
                
                # Create mask for valid distances
                far_mask = np.full(len(lon_query), False)
                far_mask[valid_query] = distances > typical_spacing
                raster_values[far_mask] = np.nan
        
        elif method == 'linear':
            # Inverse distance weighting using k nearest neighbors
            k = min(4, len(grid_points))
            distances, indices = tree.query(query_points, k=k)
            
            # Handle single point case
            if k == 1:
                raster_values[valid_query] = values_clean[indices]
            else:
                # Inverse distance weighting
                weights = 1.0 / (distances + 1e-10)
                weights /= weights.sum(axis=1, keepdims=True)
                interpolated = (values_clean[indices] * weights).sum(axis=1)
                raster_values[valid_query] = interpolated
                
                # Set pixels too far from any grid point to NaN
                if len(grid_points) > 1:
                    sample_size = min(100, len(grid_points))
                    sample_distances, _ = tree.query(grid_points[:sample_size], k=2)
                    typical_spacing = np.median(sample_distances[:, 1]) * 3
                    
                    far_mask = np.full(len(lon_query), False)
                    far_mask[valid_query] = distances[:, 0] > typical_spacing
                    raster_values[far_mask] = np.nan
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'linear'")
    
    # Reshape to 2D raster
    raster = raster_values.reshape(height_px, width_px)
    
    return raster


def to_raster_with_mask(data, grid, ax, pixel_ratio=1.0, method='nearest', 
                        mask_value=None):
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
    
    Returns
    -------
    numpy.ndarray
        2D rasterized array
    """
    if mask_value is not None:
        data = data.where(data != mask_value)
    
    return to_raster(data, grid, ax, pixel_ratio=pixel_ratio, method=method)