import numpy as np
from scipy.interpolate import interp1d

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True, fastmath=True)
def _interp_columns_numba(datai_reshaped, p_hybrid_reshaped, datao_reshaped, 
                          plevo, intyp, kxtrp):
    """
    Numba-accelerated interpolation over spatial columns.
    
    Parameters
    ----------
    datai_reshaped : ndarray (n_spatial, nlevi)
        Input data reshaped to (spatial_points, input_levels)
    p_hybrid_reshaped : ndarray (n_spatial, nlevi)
        Pressure field reshaped to (spatial_points, input_levels)
    datao_reshaped : ndarray (n_spatial, nlevo)
        Output array to fill (spatial_points, output_levels)
    plevo : ndarray (nlevo,)
        Output pressure levels
    intyp : int
        Interpolation type (1=linear, 2=log, 3=log-log)
    kxtrp : bool
        Whether to extrapolate
    
    Returns
    -------
    datao_reshaped : ndarray
        Filled output array
    """
    n_spatial = datai_reshaped.shape[0]
    nlevi = datai_reshaped.shape[1]
    nlevo = plevo.shape[0]
    
    # Process each column in parallel
    for i in prange(n_spatial):
        p_col = p_hybrid_reshaped[i, :]
        data_col = datai_reshaped[i, :]
        
        # Check for NaN values
        has_nan_data = False
        has_nan_p = False
        has_invalid_p = False
        
        for k in range(nlevi):
            if np.isnan(data_col[k]):
                has_nan_data = True
                break
            if np.isnan(p_col[k]):
                has_nan_p = True
                break
            if intyp >= 2 and p_col[k] <= 0:
                has_invalid_p = True
                break
        
        if has_nan_data or has_nan_p or has_invalid_p:
            continue
        
        # Sort by pressure (descending: high pressure/surface → low pressure/top)
        # This ensures monotonic coordinates for interpolation
        sort_idx = np.argsort(p_col)[::-1]  # Sort descending
        p_sorted = p_col[sort_idx]
        data_sorted = data_col[sort_idx]
        
        # Transform coordinates based on interpolation type
        x = np.empty(nlevi, dtype=np.float64)
        y = np.empty(nlevi, dtype=np.float64)
        x_out = np.empty(nlevo, dtype=np.float64)
        
        if intyp == 1:  # linear in p
            x = p_sorted.copy()
            y = data_sorted.copy()
            x_out = plevo.copy()
        elif intyp == 2:  # linear in log(p)
            for k in range(nlevi):
                x[k] = np.log(p_sorted[k])
                y[k] = data_sorted[k]
            for k in range(nlevo):
                x_out[k] = np.log(plevo[k])
        elif intyp == 3:  # log-log
            for k in range(nlevi):
                x[k] = np.log(p_sorted[k])
                y[k] = np.log(np.abs(data_sorted[k]))
            for k in range(nlevo):
                x_out[k] = np.log(plevo[k])
        
        # Linear interpolation for each output level
        for j in range(nlevo):
            x_interp = x_out[j]
            result = np.nan
            
            # Find bracketing points
            # x is sorted descending (high to low pressure after sorting)
            # So x[0] is highest value, x[nlevi-1] is lowest value
            if x_interp > x[0]:
                # Above highest point (higher pressure than surface)
                if kxtrp:
                    # Extrapolate using first two points
                    slope = (y[1] - y[0]) / (x[1] - x[0])
                    result = y[0] + slope * (x_interp - x[0])
            elif x_interp < x[nlevi - 1]:
                # Below lowest point (lower pressure than top)
                if kxtrp:
                    # Extrapolate using last two points
                    slope = (y[nlevi-1] - y[nlevi-2]) / (x[nlevi-1] - x[nlevi-2])
                    result = y[nlevi-1] + slope * (x_interp - x[nlevi-1])
            else:
                # Interpolate - find bracketing indices
                # Since x is descending, we need x[k] >= x_interp >= x[k+1]
                for k in range(nlevi - 1):
                    if x[k] >= x_interp >= x[k+1]:
                        # Linear interpolation
                        alpha = (x_interp - x[k]) / (x[k+1] - x[k])
                        result = y[k] + alpha * (y[k+1] - y[k])
                        break
            
            # Transform back if log-log
            if intyp == 3 and not np.isnan(result):
                result = np.exp(result)
            
            datao_reshaped[i, j] = result
    
    return datao_reshaped


def _interp_columns_scipy(datai_reshaped, p_hybrid_reshaped, datao_reshaped, 
                          plevo, intyp, kxtrp):
    """
    Scipy-based interpolation over spatial columns (fallback when numba unavailable).
    
    Same interface as numba version but uses scipy.interpolate.
    """
    n_spatial = datai_reshaped.shape[0]
    
    for i in range(n_spatial):
        p_col = p_hybrid_reshaped[i, :]
        data_col = datai_reshaped[i, :]
        
        # Skip if any values are missing or invalid
        if np.any(np.isnan(data_col)) or np.any(np.isnan(p_col)):
            continue
        
        # Skip if any pressure values are <= 0 (invalid for log interpolation)
        if intyp >= 2 and np.any(p_col <= 0):
            continue
        
        # Sort by pressure (descending: high pressure/surface → low pressure/top)
        # This ensures monotonic coordinates for interpolation
        sort_idx = np.argsort(p_col)[::-1]  # Sort descending
        p_sorted = p_col[sort_idx]
        data_sorted = data_col[sort_idx]
        
        # Apply interpolation type transformation
        if intyp == 1:  # linear in p
            x = p_sorted
            y = data_sorted
            x_out = plevo
        elif intyp == 2:  # linear in log(p)
            x = np.log(p_sorted)
            y = data_sorted
            x_out = np.log(plevo)
        elif intyp == 3:  # log-log
            x = np.log(p_sorted)
            y = np.log(np.abs(data_sorted))
            x_out = np.log(plevo)
        else:
            raise ValueError(f"Invalid intyp: {intyp}. Must be 1, 2, or 3.")
        
        # Determine extrapolation behavior
        if kxtrp:
            fill_value = 'extrapolate'
        else:
            fill_value = np.nan
        
        # Perform interpolation
        try:
            interp_func = interp1d(x, y, kind='linear', 
                                  bounds_error=False, 
                                  fill_value=fill_value)
            result = interp_func(x_out)
            
            # Transform back if log-log
            if intyp == 3:
                result = np.exp(result)
            
            datao_reshaped[i, :] = result
                
        except Exception as e:
            # If interpolation fails, leave as NaN
            continue
    
    return datao_reshaped


def vinth2p(datai, hbcofa, hbcofb, plevo, psfc, intyp, p0, ii, kxtrp, use_numba=True):
    """
    Interpolates data on hybrid coordinates to pressure levels.
    
    Python version of NCL's vinth2p function.
    Works with numpy arrays, xarray DataArrays, and dask arrays.
    Handles both structured grids (time, lev, lat, lon) and unstructured grids (n_face, lev).
    
    ALL PRESSURE VALUES MUST BE IN PASCALS (Pa).
    
    Parameters
    ----------
    datai : array_like, xarray.DataArray
        Input data on hybrid levels. 
        Structured: Shape (..., lev, lat, lon) or (..., lev, ...)
        Unstructured: Shape (n_face, lev) or (n_cells, lev)
        The level dimension should be specified by ii parameter, or will be auto-detected
        from xarray dimension names ('lev', 'level', 'z', etc.)
    hbcofa : array_like, xarray.DataArray
        Hybrid A coefficients (DIMENSIONLESS).
        Shape: (lev,)
        Note: For E3SM/CESM models, these are dimensionless coefficients
    hbcofb : array_like, xarray.DataArray
        Hybrid B coefficients (dimensionless). Shape: (lev,)
    plevo : array_like
        Output pressure levels in PASCALS (Pa). Shape: (num_levels,)
        Example: [100000, 85000, 50000, 25000, 10000] for 1000, 850, 500, 250, 100 hPa
    psfc : array_like, xarray.DataArray
        Surface pressure in PASCALS (Pa). Shape matches datai without level dimension.
    intyp : int
        Interpolation type:
        1 = linear
        2 = log
        3 = log-log
    p0 : float
        Reference pressure in PASCALS (Pa). Typically 100000 Pa (1000 hPa).
        Used in formula: p(k) = hbcofa(k) * p0 + hbcofb(k) * psfc
    ii : int or None
        Dimension index where levels are located. If None, will attempt to auto-detect
        from xarray dimension names or from matching hybrid coefficient size.
    kxtrp : bool
        If True, extrapolate below ground. If False, set to missing value.
    use_numba : bool, optional
        If True (default), use numba-accelerated interpolation when available.
        Set to False to use scipy interpolation instead.
    
    Returns
    -------
    datao : ndarray or xarray.DataArray
        Interpolated data on pressure levels. Same type and shape as datai but with
        level dimension replaced by plevo dimension.
        If input was xarray, output will be xarray with preserved attributes and coordinates.
    
    Notes
    -----
    PRESSURE FORMULA (E3SM/CESM style):
    p(k) = hbcofa(k) * p0 + hbcofb(k) * psfc
    
    where:
    - hbcofa(k) = dimensionless A coefficient
    - hbcofb(k) = dimensionless B coefficient  
    - p0 = reference pressure (Pa)
    - psfc = surface pressure (Pa)
    
    ALL PRESSURE UNITS ARE IN PASCALS (Pa):
    - plevo: Pa (not hPa)  
    - psfc: Pa (not hPa)
    - p0: Pa (not hPa)
    
    To convert from hPa to Pa, multiply by 100:
    - 1000 hPa = 100000 Pa
    - 850 hPa = 85000 Pa
    - 500 hPa = 50000 Pa
    """
    
    # Store original input for type checking and reconstruction
    input_is_xarray = False
    input_is_uxarray = False
    original_datai = datai
    lev_dim_name = None
    lev_dim_idx = None
    datai_dims = None
    datai_coords = None
    datai_attrs = None
    datai_name = None
    
    try:
        import xarray as xr
        if isinstance(datai, xr.DataArray):
            input_is_xarray = True
            datai_dims = list(datai.dims)
            datai_coords = dict(datai.coords)
            datai_attrs = dict(datai.attrs)
            datai_name = datai.name
            
            # Try to find the level dimension by name
            level_dim_names = ['lev', 'level', 'z', 'plev', 'ilev', 'height', 'vertical']
            for i, dim in enumerate(datai.dims):
                if dim.lower() in level_dim_names or 'lev' in dim.lower():
                    lev_dim_name = dim
                    lev_dim_idx = i
                    break
            
            datai = datai.values
        
        # Check for uxarray
        try:
            import uxarray as ux
            if hasattr(original_datai, 'uxgrid') or isinstance(original_datai, type(ux.UxDataArray)):
                input_is_uxarray = True
                input_is_xarray = False  # uxarray takes precedence
        except ImportError:
            pass
        
        # Handle hybrid coefficients - remove redundant time dimension if present
        if isinstance(hbcofa, xr.DataArray):
            if 'time' in hbcofa.dims and hbcofa.dims[0] == 'time':
                # Select first time step to drop time dimension
                hbcofa = hbcofa.isel(time=0)
            hbcofa = hbcofa.values
        
        if isinstance(hbcofb, xr.DataArray):
            if 'time' in hbcofb.dims and hbcofb.dims[0] == 'time':
                # Select first time step to drop time dimension
                hbcofb = hbcofb.isel(time=0)
            hbcofb = hbcofb.values
        
        if isinstance(psfc, xr.DataArray):
            psfc = psfc.values
    except ImportError:
        pass
    
    # Convert inputs to numpy arrays (handles dask arrays)
    try:
        # If dask array, compute it
        if hasattr(datai,  'compute'): datai = datai.compute()
        if hasattr(psfc,   'compute'): psfc = psfc.compute()
        if hasattr(hbcofa, 'compute'): hbcofa = hbcofa.compute()
        if hasattr(hbcofb, 'compute'): hbcofb = hbcofb.compute()
    except:
        pass
    
    datai = np.asarray(datai)
    hbcofa = np.asarray(hbcofa, dtype=np.float64).flatten()  # Ensure 1D float64
    hbcofb = np.asarray(hbcofb, dtype=np.float64).flatten()  # Ensure 1D float64
    plevo = np.atleast_1d(np.asarray(plevo, dtype=np.float64))  # Ensure 1D float64
    psfc = np.asarray(psfc)
    
    # Determine the level dimension index
    if ii is None:
        # Try to auto-detect
        if lev_dim_idx is not None:
            # Found from xarray dimension name
            ii = lev_dim_idx
        else:
            # Search for dimension that matches hybrid coefficient length
            hyb_len = len(hbcofa)
            for i, dim_size in enumerate(datai.shape):
                if dim_size == hyb_len:
                    ii = i
                    break
            
            # If still not found, use default based on number of dimensions
            if ii is None:
                if datai.ndim == 2:
                    ii = -1  # Unstructured: (n_face, lev)
                elif datai.ndim >= 3:
                    ii = -2  # Structured: (..., lev, lat, lon)
                else:
                    ii = 0  # Fallback
    
    # Move level dimension to position -1 (last position) for easier processing
    original_shape = datai.shape
    original_ii = ii
    
    if ii != -1:
        datai = np.moveaxis(datai, ii, -1)
    
    # Get dimensions
    nlevi = datai.shape[-1]  # number of input levels (now always last dimension)
    nlevo = plevo.size  # number of output levels
    
    # Verify hybrid coefficients match number of levels
    if len(hbcofa) != nlevi or len(hbcofb) != nlevi:
        raise ValueError(f"Hybrid coefficients length ({len(hbcofa)}, {len(hbcofb)}) must match number of levels ({nlevi})")
    
    # Ensure psfc matches datai shape (without level dimension)
    expected_psfc_shape = list(datai.shape[:-1])  # All dims except level
    
    # Handle common case: psfc has time dimension but datai doesn't
    if psfc.ndim == datai.ndim and psfc.shape != tuple(expected_psfc_shape):
        # Check if removing first dimension of psfc would make shapes match
        if psfc.shape[1:] == tuple(expected_psfc_shape):
            print(f"Warning: psfc has shape {psfc.shape} but datai expects {tuple(expected_psfc_shape)}.")
            print(f"         Automatically selecting first time slice: psfc[0, ...]")
            psfc = psfc[0, ...]
    
    if psfc.shape != tuple(expected_psfc_shape):
        # Try to squeeze or reshape psfc to match
        if psfc.size == np.prod(expected_psfc_shape):
            psfc = psfc.reshape(expected_psfc_shape)
        else:
            raise ValueError(
                f"Surface pressure shape {psfc.shape} doesn't match expected shape {tuple(expected_psfc_shape)}.\n"
                f"Data shape is {datai.shape} (level at position {original_ii}).\n"
                f"Expected psfc to have all dimensions except level.\n"
                f"If psfc has a time dimension, select the appropriate time slice first:\n"
                f"  psfc_slice = psfc[time_index, ...] or psfc.isel(time=time_index)"
            )
    
    # Calculate pressure at each hybrid level using E3SM/CESM formula
    # p(k) = hbcofa(k) * p0 + hbcofb(k) * psfc
    
    # Build the shape for broadcasting hybrid coefficients
    # Target shape should match datai shape with coefficients on the level dimension
    coef_shape = [1] * datai.ndim
    coef_shape[-1] = nlevi  # Set level dimension (now last)
    
    hbcofa_broadcast = hbcofa.reshape(coef_shape)
    hbcofb_broadcast = hbcofb.reshape(coef_shape)
    
    # Build shape for psfc broadcasting
    # psfc should have all dimensions of datai except the level dimension
    # Add a singleton dimension for the level axis at the end
    psfc_expanded = np.expand_dims(psfc, axis=-1)
    
    # Now broadcast to compute pressure field using E3SM/CESM formula
    p_hybrid = hbcofa_broadcast * p0 + hbcofb_broadcast * psfc_expanded
    
    # Initialize output array
    output_shape = list(datai.shape)
    output_shape[-1] = nlevo  # Replace level dimension size
    datao = np.full(output_shape, np.nan)
    
    # For efficiency, reshape to (n_spatial, nlev)
    # where n_spatial is product of all dimensions except level
    n_spatial = int(np.prod(datai.shape[:-1])) if len(datai.shape) > 1 else 1
    
    # Reshape data and pressure - ensure float64 for numba
    datai_reshaped = datai.reshape(n_spatial, nlevi).astype(np.float64)
    p_hybrid_reshaped = p_hybrid.reshape(n_spatial, nlevi).astype(np.float64)
    datao_reshaped = datao.reshape(n_spatial, nlevo)
    
    # Choose interpolation method
    if use_numba and NUMBA_AVAILABLE:
        # Use numba-accelerated version
        datao_reshaped = _interp_columns_numba(datai_reshaped, p_hybrid_reshaped, 
                                               datao_reshaped, plevo, intyp, kxtrp)
    else:
        # Use scipy version
        datao_reshaped = _interp_columns_scipy(datai_reshaped, p_hybrid_reshaped, 
                                               datao_reshaped, plevo, intyp, kxtrp)
    
    # Reshape back to original structure
    datao = datao_reshaped.reshape(output_shape)
    
    # Move level dimension back to original position if needed
    if original_ii != -1:
        datao = np.moveaxis(datao, -1, original_ii)
    
    # Reconstruct xarray/uxarray object if input was xarray/uxarray
    if input_is_xarray or input_is_uxarray:
        try:
            import xarray as xr
            
            # Create new coordinates with updated pressure levels
            new_coords = {}
            new_dims = []
            
            for i, dim in enumerate(datai_dims):
                if dim == lev_dim_name:
                    # Replace level dimension with pressure levels
                    new_dim_name = 'plev'  # Use standard name for pressure levels
                    new_dims.append(new_dim_name)
                    new_coords[new_dim_name] = plevo
                else:
                    new_dims.append(dim)
                    if dim in datai_coords:
                        new_coords[dim] = datai_coords[dim]
            
            # Create xarray DataArray
            datao_xr = xr.DataArray(
                datao,
                dims=new_dims,
                coords=new_coords,
                attrs=datai_attrs,
                name=datai_name
            )
            
            # Add metadata about interpolation
            datao_xr.attrs['interpolation_method'] = {1: 'linear', 2: 'log', 3: 'loglog'}[intyp]
            datao_xr.attrs['vertical_coordinate'] = 'pressure'
            datao_xr.attrs['pressure_units'] = 'Pa'
            
            # If input was uxarray, convert back to uxarray
            if input_is_uxarray:
                try:
                    import uxarray as ux
                    # Reconstruct uxarray DataArray with the grid from original
                    if hasattr(original_datai, 'uxgrid'):
                        datao = ux.UxDataArray(
                            datao_xr,
                            uxgrid=original_datai.uxgrid
                        )
                    else:
                        datao = datao_xr
                except ImportError:
                    datao = datao_xr
            else:
                datao = datao_xr
                
        except ImportError:
            # If xarray import fails, just return numpy array
            pass
    
    return datao


def vinth2p_simple(datai, hbcofa, hbcofb, plevo, psfc, p0=100000.0, 
                   interp_type='log', extrapolate=False, use_numba=True):
    """
    Simplified interface for vinth2p with common defaults.
    
    ALL PRESSURE VALUES MUST BE IN PASCALS (Pa).
    
    Parameters
    ----------
    datai : array_like, xarray.DataArray, uxarray.UxDataArray
        Input data. 
        Structured: Shape (time, lev, lat, lon) or (lev, lat, lon)
        Unstructured: Shape (n_face, lev) or (time, n_face, lev)
    hbcofa : array_like, xarray.DataArray
        Hybrid A coefficients (DIMENSIONLESS for E3SM/CESM models)
    hbcofb : array_like, xarray.DataArray
        Hybrid B coefficients (dimensionless)
    plevo : array_like
        Output pressure levels in PASCALS (Pa).
        Example: [100000, 85000, 50000] for 1000, 850, 500 hPa levels
        Can also use: [1000] for 1000 Pa (10 hPa)
    psfc : array_like, xarray.DataArray
        Surface pressure in PASCALS (Pa). Shape matches datai without level dimension
    p0 : float, optional
        Reference pressure in PASCALS (Pa) (default: 100000.0 Pa = 1000 hPa)
    interp_type : str, optional
        'linear', 'log', or 'loglog' (default: 'log')
    extrapolate : bool, optional
        Whether to extrapolate below surface (default: False)
    use_numba : bool, optional
        If True (default), use numba-accelerated interpolation when available.
    
    Returns
    -------
    datao : ndarray, xarray.DataArray, or uxarray.UxDataArray
        Interpolated data on pressure levels. Returns same type as input.
        - If input was xarray: returns xarray with preserved attributes and new 'plev' dimension
        - If input was uxarray: returns uxarray with preserved grid and new 'plev' dimension
        - If input was numpy: returns numpy array
    
    Notes
    -----
    PRESSURE FORMULA (E3SM/CESM style):
    p(k) = hbcofa(k) * p0 + hbcofb(k) * psfc
    
    ALL PRESSURE UNITS ARE IN PASCALS (Pa):
    - To convert hPa to Pa: multiply by 100
    - Example: plevo = [100000, 85000, 50000, 25000, 10000]  # 1000, 850, 500, 250, 100 hPa
    - Example: plevo = [1000] for 10 hPa level
    
    Common pressure levels in Pa:
    - Surface:     ~100000 Pa (1000 hPa)
    - 850 hPa:      85000 Pa
    - 500 hPa:      50000 Pa  
    - 250 hPa:      25000 Pa
    - 100 hPa:      10000 Pa
    - 10 hPa:        1000 Pa
    """
    
    # Map string types to integer codes
    type_map = {'linear': 1, 'log': 2, 'loglog': 3}
    intyp = type_map.get(interp_type.lower(), 2)
    
    # Ensure plevo is array-like
    plevo = np.atleast_1d(np.asarray(plevo, dtype=np.float64))
    
    # Let vinth2p auto-detect the level dimension
    # Pass None as ii to trigger auto-detection
    return vinth2p(datai, hbcofa, hbcofb, plevo, psfc, intyp, p0, None, 
                   extrapolate, use_numba=use_numba)


if __name__ == "__main__":
    import time
    
    # Example usage
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)
    
    print("\nExample 1: Structured grid (lev, lat, lon)")
    print("=" * 60)
    
    # Create sample data
    nlev_in = 10
    nlev_out = 5
    nlat = 20
    nlon = 30
    
    # Hybrid coefficients (example values in Pa)
    hbcofa = np.linspace(0, 5000, nlev_in)  # Pa
    hbcofb = np.linspace(0, 1, nlev_in)  # dimensionless
    
    # Output pressure levels in Pa (not hPa!)
    plevo = np.array([100000, 85000, 50000, 25000, 10000])  # Pa (1000, 850, 500, 250, 100 hPa)
    
    # Surface pressure in Pa
    psfc = np.random.uniform(95000, 105000, (nlat, nlon))  # Pa
    
    # Sample data on hybrid levels
    datai = np.random.randn(nlev_in, nlat, nlon)
    
    # Interpolate using simplified interface
    t0 = time.time()
    datao = vinth2p_simple(datai, hbcofa, hbcofb, plevo, psfc, 
                          p0=100000.0, interp_type='log')
    t1 = time.time()
    
    print(f"Input shape:  {datai.shape}")
    print(f"Output shape: {datao.shape}")
    print(f"Output levels: {plevo} Pa ({plevo/100} hPa)")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # Test with 4D data
    print("\n" + "=" * 60)
    print("Example 2: Structured grid with time (time, lev, lat, lon)")
    print("=" * 60)
    ntime = 5
    datai_4d = np.random.randn(ntime, nlev_in, nlat, nlon)
    psfc_4d = np.random.uniform(95000, 105000, (ntime, nlat, nlon))
    
    t0 = time.time()
    datao_4d = vinth2p_simple(datai_4d, hbcofa, hbcofb, plevo, psfc_4d, 
                             p0=100000.0, interp_type='log')
    t1 = time.time()
    
    print(f"Input shape:  {datai_4d.shape}")
    print(f"Output shape: {datao_4d.shape}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # Test with unstructured grid
    print("\n" + "=" * 60)
    print("Example 3: Unstructured grid (n_face, lev)")
    print("=" * 60)
    n_face = 10000
    datai_unstr = np.random.randn(n_face, nlev_in)
    psfc_unstr = np.random.uniform(95000, 105000, (n_face,))
    
    t0 = time.time()
    datao_unstr = vinth2p_simple(datai_unstr, hbcofa, hbcofb, plevo, psfc_unstr,
                                 p0=100000.0, interp_type='log')
    t1 = time.time()
    
    print(f"Input shape:  {datai_unstr.shape}")
    print(f"Output shape: {datao_unstr.shape}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # Test with hybrid coefficients that have time dimension
    print("\n" + "=" * 60)
    print("Example 4: Hybrid coefficients with redundant time dimension")
    print("=" * 60)
    
    try:
        import xarray as xr
        
        # Create sample data with xarray
        ntime_coef = 10
        hbcofa_with_time = xr.DataArray(
            np.broadcast_to(hbcofa, (ntime_coef, nlev_in)),
            dims=['time', 'lev'],
            coords={'time': np.arange(ntime_coef), 'lev': np.arange(nlev_in)}
        )
        hbcofb_with_time = xr.DataArray(
            np.broadcast_to(hbcofb, (ntime_coef, nlev_in)),
            dims=['time', 'lev'],
            coords={'time': np.arange(ntime_coef), 'lev': np.arange(nlev_in)}
        )
        
        # Test data
        datai_test = xr.DataArray(
            np.random.randn(n_face, nlev_in),
            dims=['n_face', 'lev'],
            coords={'lev': np.arange(nlev_in)}
        )
        psfc_test = np.random.uniform(95000, 105000, (n_face,))
        
        t0 = time.time()
        datao_test = vinth2p_simple(datai_test, hbcofa_with_time, hbcofb_with_time, 
                                   plevo, psfc_test, p0=100000.0)
        t1 = time.time()
        
        print(f"Hybrid coef shape: {hbcofa_with_time.shape} (with time dimension)")
        print(f"Input shape:  {datai_test.shape}")
        print(f"Output shape: {datao_test.shape}")
        print(f"Time: {(t1-t0)*1000:.2f} ms")
        print("✓ Successfully handled hybrid coefficients with time dimension")
        
    except ImportError:
        print("xarray not available - skipping this test")
    
    print("\n" + "=" * 60)
    print("UNIT CONVERSION REFERENCE")
    print("=" * 60)
    print("PRESSURE FORMULA (E3SM/CESM):")
    print("  p(k) = hbcofa(k) * p0 + hbcofb(k) * psfc")
    print("  where hbcofa and hbcofb are DIMENSIONLESS")
    print("\nALL PRESSURE VALUES MUST BE IN PASCALS (Pa):")
    print("  1000 hPa = 100000 Pa")
    print("   850 hPa =  85000 Pa")
    print("   500 hPa =  50000 Pa")
    print("   250 hPa =  25000 Pa")
    print("   100 hPa =  10000 Pa")
    print("    10 hPa =   1000 Pa")
    print("\nTo convert from hPa to Pa: multiply by 100")
    print("To convert from Pa to hPa: divide by 100")
    
    # Performance comparison if numba available
    if NUMBA_AVAILABLE:
        print("\n" + "=" * 60)
        print("Performance comparison: Large unstructured grid")
        print("=" * 60)
        
        n_face = 100000
        datai_large = np.random.randn(n_face, nlev_in)
        psfc_large = np.random.uniform(95000, 105000, (n_face,))
        
        # Numba version
        t0 = time.time()
        datao_numba = vinth2p_simple(datai_large, hbcofa, hbcofb, plevo, psfc_large,
                                     p0=100000.0, interp_type='log', use_numba=True)
        t1 = time.time()
        time_numba = t1 - t0
        
        # Scipy version
        t0 = time.time()
        datao_scipy = vinth2p_simple(datai_large, hbcofa, hbcofb, plevo, psfc_large,
                                     p0=100000.0, interp_type='log')
        t1 = time.time()
        time_scipy = t1 - t0
        
        print(f"Grid size: {n_face} cells, {nlev_in} input levels, {nlev_out} output levels")
        print(f"Numba time:  {time_numba:.3f} s")
        print(f"Scipy time:  {time_scipy:.3f} s")
        print(f"Speedup:     {time_scipy/time_numba:.1f}x")
