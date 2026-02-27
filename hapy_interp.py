from hapy_common import *
#---------------------------------------------------------------------------------------------------
plev_default = np.array([10,30,50,75,100,125,150,200,250,300,350,400,450,500,
                                 550,600,650,700,750,800,825,850,875,900,925,950,975,1000])
# plev_default = np.array([1,2,4,6,10,30,50,100,150,200,300,400,500,600,700,800,850,925,950])
# plev_default = np.array([5,10,30,50,100,150,200,300,400,500,600,700,800,850,925,975,1000])
#---------------------------------------------------------------------------------------------------
def interpolate_to_pressure(ds,data_mlev,plev=None,interp_type=2,extrap_flag=False):
    global plev_default
    exit('ERROR - interpolate_to_pressure() needs a replacement for ngl.vinth2p()!')
    if plev is None: plev = plev_default
    #----------------------------------------------------------------------------
    hya = ds['hyam'].isel(time=0,missing_dims='ignore').values
    hyb = ds['hybm'].isel(time=0,missing_dims='ignore').values
    #----------------------------------------------------------------------------
    # Create empty array with new lev dim
    data_plev = xr.full_like( data_mlev.isel(lev=0,drop=True), np.nan )
    data_plev = data_plev.expand_dims(dim={'lev':plev}, axis=data_mlev.get_axis_num('lev'))
    #----------------------------------------------------------------------------
    # Add dummy dimension
    data_mlev = data_mlev.expand_dims(dim='dummy',axis=len(data_mlev.dims))
    #----------------------------------------------------------------------------
    PS_dum = ds['ps'].expand_dims(dim='dummy',axis=len(ds['ps'].dims))
    P0 = xr.DataArray(1e3)
    data_mlev = data_mlev.transpose('time','lev','ncol','dummy')
    data_plev = data_plev.transpose('time','lev','ncol')
    # Do the interpolation in chunks
    c1 = 0
    num_chunk = len(data_plev.chunksizes['ncol'])
    #----------------------------------------------------------------------------
    for c,sz in enumerate(data_plev.chunksizes['ncol']):
        c2 = c1+sz
        # data_plev[:,:,c1:c2] = ngl.vinth2p( data_mlev[:,:,c1:c2,:].values, \
        #                                     hya, hyb, plev, PS_dum[:,c1:c2,:], \
        #                                     interp_type, P0, 1, extrap_flag)[:,:,:,0]
        c1 = c2
    data_plev = xr.DataArray( np.ma.masked_values( data_plev ,1e30), coords=data_plev.coords )
    return data_plev
#---------------------------------------------------------------------------------------------------
def interp_to_height(
    da: xr.DataArray,
    zgeo: xr.DataArray,
    target_heights: ArrayLike,
    lev_dim: str = "lev",
    height_dim: str = "height",
    extrapolate: bool = False,
) -> xr.DataArray:
    """
    Interpolate a DataArray from hybrid sigma-pressure levels to constant
    height levels using variable input heights.

    The input may have any shape as long as it includes a vertical dimension
    named `lev_dim`.  Common cases:

        (lev,)             – single column, no ncol dimension
        (ncol, lev)        – standard E3SM unstructured layout
        (time, ncol, lev)  – time-varying unstructured data

    Parameters
    ----------
    da : xr.DataArray
        Input field containing dimension `lev_dim`.
    zgeo : xr.DataArray
        Mid-point geometric height [m].  Must share `lev_dim` (and the ncol
        dimension, if present) with `da`.
    target_heights : array-like
        Target height levels [m], ascending or descending.
    lev_dim : str
        Name of the vertical dimension in `da` and `zgeo`.
    height_dim : str
        Name of the output vertical dimension.
    extrapolate : bool
        If True, linearly extrapolate beyond each column's height range.
        If False (default), out-of-range values are NaN.

    Returns
    -------
    xr.DataArray
        Interpolated field with `lev_dim` replaced by `height_dim`.
    """
    target_heights = np.asarray(target_heights, dtype=np.float64)
    n_heights = len(target_heights)

    # ------------------------------------------------------------------
    # 1. Align zgeo to da's dimension order, then move lev to last axis.
    # ------------------------------------------------------------------
    zgeo = zgeo.transpose(*[d for d in da.dims if d in zgeo.dims])

    lev_ax = da.dims.index(lev_dim)
    other_dims = [d for d in da.dims if d != lev_dim]  # dims that survive

    # Move lev to last position for uniform indexing
    da_vals  = np.moveaxis(da.values,   lev_ax, -1)   # (..., nlev)
    z_vals   = np.moveaxis(zgeo.values, zgeo.dims.index(lev_dim), -1)

    # ------------------------------------------------------------------
    # 2. Broadcast zgeo to match da's non-lev shape if needed (e.g. time).
    # ------------------------------------------------------------------
    if da_vals.shape != z_vals.shape:
        z_vals = np.broadcast_to(z_vals, da_vals.shape).copy()

    orig_shape = da_vals.shape          # (..., nlev)
    nlev       = orig_shape[-1]
    col_shape  = orig_shape[:-1]        # shape of all non-lev dims

    # Flatten all non-lev dims into one "columns" axis
    n_cols    = int(np.prod(col_shape)) if col_shape else 1
    da_flat   = da_vals.reshape(n_cols, nlev)
    z_flat    = z_vals.reshape(n_cols, nlev)

    # ------------------------------------------------------------------
    # 3. Determine height ordering from the first column.
    # ------------------------------------------------------------------
    z_increasing = z_flat[0, -1] > z_flat[0, 0]   # True → bottom-to-top

    # ------------------------------------------------------------------
    # 4. Interpolate each column.
    # ------------------------------------------------------------------
    out_flat = np.full((n_cols, n_heights), np.nan, dtype=np.float64)

    for i in range(n_cols):
        z_col = z_flat[i]
        v_col = da_flat[i]

        if not z_increasing:          # flip to ascending height order
            z_col = z_col[::-1]
            v_col = v_col[::-1]

        if extrapolate:
            out_flat[i] = np.interp(target_heights, z_col, v_col)
            lo = target_heights < z_col[0]
            hi = target_heights > z_col[-1]
            if lo.any():
                slope = (v_col[1] - v_col[0]) / (z_col[1] - z_col[0])
                out_flat[i, lo] = v_col[0] + slope * (target_heights[lo] - z_col[0])
            if hi.any():
                slope = (v_col[-1] - v_col[-2]) / (z_col[-1] - z_col[-2])
                out_flat[i, hi] = v_col[-1] + slope * (target_heights[hi] - z_col[-1])
        else:
            out_flat[i] = np.interp(target_heights, z_col, v_col,
                                    left=np.nan, right=np.nan)

    # ------------------------------------------------------------------
    # 5. Restore original non-lev shape and reinsert height axis.
    # ------------------------------------------------------------------
    out = out_flat.reshape(*col_shape, n_heights) if col_shape else out_flat.reshape(n_heights)

    # ------------------------------------------------------------------
    # 6. Rebuild DataArray with correct dims/coords.
    # ------------------------------------------------------------------
    coords = {k: v for k, v in da.coords.items() if lev_dim not in v.dims}
    coords[height_dim] = xr.DataArray(
        target_heights, dims=[height_dim],
        attrs={"units": "m", "long_name": "geometric height"},
    )

    out_dims = [d if d != lev_dim else height_dim for d in da.dims]

    return xr.DataArray(
        out,
        dims=out_dims,
        coords=coords,
        attrs={**da.attrs, "note": "Interpolated to constant height levels"},
        name=da.name,
    )
#---------------------------------------------------------------------------------------------------
