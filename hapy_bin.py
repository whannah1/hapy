from hapy_common import *
#---------------------------------------------------------------------------------------------------
# Binning Routines
#---------------------------------------------------------------------------------------------------
def _bin_coord_manual(bin_min, bin_max, bin_spc):
    """ Build linearly spaced bin centers and edges for manual mode.
    Returns (nbin, bin_coord, bin_edges) where bin_edges is (nbin, 2). """
    nbin = np.round((bin_max - bin_min + bin_spc) / bin_spc).astype(int)
    centers = np.linspace(bin_min, bin_max, nbin)
    edges = np.column_stack([centers - bin_spc/2., centers + bin_spc/2.])
    return nbin, xr.DataArray(centers), edges
#---------------------------------------------------------------------------------------------------
def _bin_coord_explicit(bins):
    """ Build bin centers and edges from explicit bin edge values.
    Returns (nbin, bin_coord, bin_edges) where bin_edges is (nbin, 2). """
    bins = xr.DataArray(bins)
    nbin = len(bins) - 1
    centers = (bins[0:nbin] + bins[1:nbin+1]) / 2.
    edges = np.column_stack([bins[0:nbin].values, bins[1:nbin+1].values])
    return nbin, centers, edges
#---------------------------------------------------------------------------------------------------
def _bin_coord_log(bin_min, bin_spc, bin_spc_log, nbin_log):
    """ Build logarithmically spaced bin centers and edges where each bin is
    wider than the previous by bin_spc_log [%].
    Returns (nbin, bin_coord, bin_edges) where bin_edges is (nbin, 2). """
    wid = np.zeros(nbin_log)
    ctr = np.zeros(nbin_log)
    ctr[0] = bin_min
    wid[0] = bin_spc
    for b in range(1, nbin_log):
        wid[b] = wid[b-1] * (1. + bin_spc_log/1e2)
        ctr[b] = ctr[b-1] + wid[b-1]/2. + wid[b]/2.
    edges = np.column_stack([ctr - wid/2., ctr + wid/2.])
    return nbin_log, xr.DataArray(ctr), edges
#---------------------------------------------------------------------------------------------------
def bin_YbyX(Vy, Vx, bins=[], bin_min=0, bin_max=1, bin_spc=1, bin_spc_log=20, nbin_log=2,
             bin_mode="manual", verbose=False, wgt=None,
             keep_time=False, keep_lev=False, method='mean'):
    """ Average Vy into bins of Vx values.
    Manual mode uses min/max/spc to define equally spaced bins.
    Explicit mode takes a list of bin edge values for irregular spacing (e.g. logarithmic).
    Log mode builds logarithmically spaced bins that increase by bin_spc_log [%] each step. """
    #----------------------------------------------------------------------------
    # build bin coordinates and edges from the selected mode
    if   bin_mode == 'manual':   nbin, bin_coord, bin_edges = _bin_coord_manual(bin_min, bin_max, bin_spc)
    elif bin_mode == 'explicit': nbin, bin_coord, bin_edges = _bin_coord_explicit(bins)
    elif bin_mode == 'log':      nbin, bin_coord, bin_edges = _bin_coord_log(bin_min, bin_spc, bin_spc_log, nbin_log)
    else:
        raise ValueError(f"Unknown bin_mode '{bin_mode}'. Expected 'manual', 'explicit', or 'log'.")
    #----------------------------------------------------------------------------
    # check for time and vertical dimensions
    valid_vert_dims = ['lev','plev','pressure_level']
    valid_time_dims = ['time']
    has_vert, vert_name = False, None
    has_time, time_name = False, None
    for d in valid_vert_dims:
        if d in Vy.dims:
            has_vert = True
            vert_name = d
    for d in valid_time_dims:
        if d in Vy.dims:
            has_time = True
            time_name = d
    #----------------------------------------------------------------------------

    nvert = len(Vy[vert_name]) if has_vert else 1
    ntime = len(Vy[time_name]) if has_time else 1
    if ntime == 1: keep_time = False

    lev_chk = has_vert and nvert>1 and keep_lev
    #----------------------------------------------------------------------------
    # determine output shape based on dimensions to retain
    if lev_chk:
        shape = (nbin, nvert)
        dims  = ['bin', vert_name]
        coord = [('bin', bin_coord.values), (vert_name, Vy[vert_name].values)]
    elif has_time and keep_time:
        shape = (nbin, ntime)
        dims  = ['bin', time_name]
        coord = [('bin', bin_coord.values), (time_name, Vy[time_name].values)]
    else:
        shape = (nbin,)
        dims  = ['bin']
        coord = [('bin', bin_coord.values)]

    if verbose:
        print(f'\n  shape: {shape}\n  dtype: {Vy.dtype}\n  coord: {coord}\n  dims : {dims}\n')
    #----------------------------------------------------------------------------
    # initialise output arrays
    fill    = np.full(shape, np.nan, dtype=Vy.dtype)
    bin_val = xr.DataArray(fill.copy(),                     coords=coord)#, dims=dims)
    bin_std = xr.DataArray(fill.copy(),                     coords=coord)#, dims=dims)
    bin_cnt = xr.DataArray(np.zeros(shape, dtype=Vy.dtype), coords=coord)#, dims=dims)
    #----------------------------------------------------------------------------
    # broadcast weights to match the shape and dimension order of Vy
    if wgt is not None: wgt = wgt.broadcast_like(Vy)
    #---------------------------------------------------------------------------
    # Determine which dimensions to average over

    # Step 1: Does the data have a vertical level dimension?
    if lev_chk:
        # Step 2: Does the data also have a time dimension?
        if has_time:
            # Average over both time and horizontal columns
            avg_dims = [time_name, 'ncol']
        else:
            # Average over horizontal columns only
            avg_dims = ['ncol']
    else:
        # No level dimension — averaging dims are not applicable
        avg_dims = None
    #----------------------------------------------------------------------------
    # pre-load data and build finite-value mask for Vx
    Vy.load()
    val_chk = np.isfinite(Vx.values)
    #----------------------------------------------------------------------------
    # loop over bins
    for b in range(nbin):
        bin_bot, bin_top = bin_edges[b]
        Vx_safe   = np.where(val_chk, Vx.values, bin_bot - 1e3)
        condition = xr.DataArray((Vx_safe >= bin_bot) & (Vx_safe < bin_top), coords=Vx.coords)
        if not condition.values.any():
            continue
        #-----------------------------------------------------------------------
        if lev_chk:
            if wgt is None:
                bin_val[b, :] = Vy.where(condition, drop=True).mean(dim=avg_dims, skipna=True)
            else:
                weighted = ( (Vy*wgt).where(condition, drop=True).sum(dim='ncol', skipna=True)
                            /    wgt .where(condition, drop=True).sum(dim='ncol', skipna=True) )
                if time_name in Vy.dims:
                    weighted = weighted.mean(dim=time_name, skipna=True)
                bin_val[b, :] = weighted
            bin_std[b, :] = Vy.where(condition, drop=True).std( dim=avg_dims, skipna=True)
            bin_cnt[b, :] = Vy.where(condition, drop=True).count(dim=avg_dims)
        #-----------------------------------------------------------------------
        elif keep_time and time_name in Vy.dims:
            bin_val[b, :] = ( (Vy*wgt).where(condition, drop=True).sum(dim='ncol', skipna=True)
                             /    wgt .where(condition, drop=True).sum(dim='ncol', skipna=True) )
            bin_cnt[b]    = condition.values.sum()
        #-----------------------------------------------------------------------
        else:
            if method == 'max':
                bin_val[b] = Vy.where(condition).max(skipna=True)
            elif method == 'std':
                bin_val[b] = Vy.where(condition).std(skipna=True)
            else:  # mean
                if wgt is None:
                    bin_val[b] = Vy.where(condition).mean(skipna=True)
                else:
                    bin_val[b] = ( (Vy*wgt).where(condition).sum(skipna=True)
                                  /    wgt .where(condition).sum(skipna=True) )
            bin_std[b] = Vy.where(condition).std(skipna=True)
            bin_cnt[b] = condition.values.sum()
        #-----------------------------------------------------------------------
        if verbose and not lev_chk:
            print(f'  b: {b:8.0f}  cnt: {bin_cnt[b].values:8.0f}  val: {bin_val[b].values:12.4e}')
    #----------------------------------------------------------------------------
    # # make sure bin coord is correct
    # bin_val.coords['bin'] = bin_coord
    # bin_std.coords['bin'] = bin_coord
    # bin_cnt.coords['bin'] = bin_coord
    #----------------------------------------------------------------------------
    # assemble output dataset
    bin_ds = xr.Dataset({ 'bin_val': bin_val,
                          'bin_std': bin_std,
                          'bin_cnt': bin_cnt,
                          'bin_pct': bin_cnt / bin_cnt.sum() * 1e2 })
    if lev_chk:
        bin_ds.coords[vert_name] = xr.DataArray(Vy[vert_name])
    #----------------------------------------------------------------------------
    return bin_ds
#---------------------------------------------------------------------------------------------------
def bin_ZbyYX(Vz, Vy, Vx, binsy, binsx, opt):
    """ """
#---------------------------------------------------------------------------------------------------
