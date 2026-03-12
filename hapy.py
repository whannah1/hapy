from hapy_common import *
import hapy_constants as constants
from hapy_stat import *
from hapy_mach import *
from hapy_sphere import *
from hapy_sat import *
from hapy_filter import *
from hapy_bin import *
from hapy_interp import *
from hapy_vinth2p import *
from hapy_raster import *
#---------------------------------------------------------------------------------------------------
xr.set_options(use_new_combine_kwarg_defaults=True)
#---------------------------------------------------------------------------------------------------
# below are misc methods that aren't easily classified into other files
#---------------------------------------------------------------------------------------------------
def trim_png(fig_file,verbose=True,root=None):
    """ use imagemagick to crop white space from png file """
    if '.png' not in fig_file: fig_file = fig_file+'.png'
    # if root is not None:
    #    fig_file = fig_file.replace(root+'/','')
    if os.path.isfile(fig_file):
        cmd = "magick convert -trim +repage "+fig_file+"   "+fig_file
        os.system(cmd)
        if verbose: print(f'\n{fig_file}\n')
    else:
        raise FileNotFoundError(f'\ntrim_png(): {fig_file} does not exist?!\n')
#---------------------------------------------------------------------------------------------------
def print_time_length(ds_or_time, indent='    ', print_msg=True, color=None):
    """
    Print (or return) a string describing the temporal extent of a dataset or time coordinate.

    Parameters
    ----------
    ds_or_time : xr.Dataset, xr.DataArray, or time-coordinate DataArray
        If a Dataset is passed, time bounds variables (time_bnds, time_bounds, time_bnd)
        are used when available for accurate span calculation regardless of E3SM
        end-of-period time conventions. A bare time coordinate DataArray is also accepted.
    indent : str
        Prefix string for the printed line.
    print_msg : bool
        If True, print the message and return None. If False, return the message string.
    color : str or None
        ANSI escape code string to wrap the message in (e.g. tclr.RED). tclr.END is
        appended automatically.
    """
    #----------------------------------------------------------------------------
    # extract time coordinate and optional bounds from input
    time_bnds_da = None
    if isinstance(ds_or_time, xr.Dataset):
        ds = ds_or_time
        if 'time' not in ds.coords and 'time' not in ds.dims:
            msg = indent + 'Time span   : no time dimension'
            if color is not None: msg = f'{color}{msg}{tclr.END}'
            if print_msg: print(msg); return
            return msg
        time = ds['time']
        for bname in ('time_bnds', 'time_bounds', 'time_bnd'):
            if bname in ds:
                time_bnds_da = ds[bname]
                break
    elif isinstance(ds_or_time, xr.DataArray):
        da = ds_or_time
        if 'time' in da.coords and da.name != 'time':
            time = da['time']
        elif np.issubdtype(da.dtype, np.datetime64):
            time = da
        else:
            msg = indent + 'Time span   : no time dimension'
            if color is not None: msg = f'{color}{msg}{tclr.END}'
            if print_msg: print(msg); return
            return msg
    else:
        msg = indent + 'Time span   : no time dimension'
        if color is not None: msg = f'{color}{msg}{tclr.END}'
        if print_msg: print(msg); return
        return msg
    #----------------------------------------------------------------------------
    # helper: format a timedelta64[D] into a multi-unit human-readable string
    def _fmt_td(td_dy):
        s = str(td_dy)
        mn = td_dy.astype('timedelta64[M]')
        yr = mn.astype('timedelta64[Y]')
        if mn >= np.timedelta64(2, 'M'):
            s += f'  /  {mn}'
        if yr >= np.timedelta64(2, 'Y'):
            s += f'  /  {yr}'
        return s
    #----------------------------------------------------------------------------
    # handle single time value
    if len(time) <= 1:
        if time_bnds_da is not None and len(time_bnds_da) >= 1:
            span_dy = (time_bnds_da[0, 1] - time_bnds_da[0, 0]).values.astype('timedelta64[D]')
            msg = indent + f'Time span   : {_fmt_td(span_dy)}  (single period, from time_bnds)'
        else:
            msg = indent + 'Time span   : single time value, no bounds available'
        if color is not None: msg = f'{color}{msg}{tclr.END}'
        if print_msg: print(msg); return
        return msg
    #----------------------------------------------------------------------------
    # compute span: prefer bounds if available, otherwise fall back to arithmetic
    if time_bnds_da is not None:
        span_dy = (time_bnds_da[-1, 1] - time_bnds_da[0, 0]).values.astype('timedelta64[D]')
        msg = indent + f'Time span   : {_fmt_td(span_dy)}  (from time_bnds)'
    else:
        dt = time[1] - time[0]
        span_dy = (time[-1] - time[0] + dt).values.astype('timedelta64[D]')
        msg = indent + f'Time span   : {_fmt_td(span_dy)}'
    if color is not None: msg = f'{color}{msg}{tclr.END}'
    if print_msg: print(msg); return
    return msg

#---------------------------------------------------------------------------------------------------
def check_invalid_values(uxds, variables=None, check_nan=True, check_inf=True, 
                        check_range=None, verbose=True):
    """
    Check for invalid values in a uxarray dataset.
    
    Parameters:
    -----------
    uxds : uxarray.UxDataset
        The uxarray dataset to check
    variables : list, optional
        List of variable names to check. If None, checks all data variables
    check_nan : bool, default=True
        Check for NaN values
    check_inf : bool, default=True
        Check for infinite values
    check_range : dict, optional
        Dictionary of {variable_name: (min, max)} for valid ranges
    verbose : bool, default=True
        Print detailed information about invalid values found
        
    Returns:
    --------
    dict : Summary of invalid values found for each variable
    """
    
    if variables is None:
        variables = list(uxds.data_vars)
    
    results = {}
    
    for var in variables:
        if var not in uxds.data_vars:
            print(f"Warning: Variable '{var}' not found in dataset")
            continue
            
        data = uxds[var]
        invalid_info = {
            'total_values': data.size,
            'nan_count': 0,
            'inf_count': 0,
            'out_of_range_count': 0,
            'valid_count': 0
        }
        
        # Check for NaN
        if check_nan:
            nan_mask = np.isnan(data.values)
            invalid_info['nan_count'] = int(np.sum(nan_mask))
        
        # Check for inf
        if check_inf:
            inf_mask = np.isinf(data.values)
            invalid_info['inf_count'] = int(np.sum(inf_mask))
        
        # Check valid range
        if check_range and var in check_range:
            min_val, max_val = check_range[var]
            finite_data = data.values[np.isfinite(data.values)]
            out_of_range = np.sum((finite_data < min_val) | (finite_data > max_val))
            invalid_info['out_of_range_count'] = int(out_of_range)
            invalid_info['specified_range'] = (min_val, max_val)
            if len(finite_data) > 0:
                invalid_info['actual_range'] = (float(np.min(finite_data)), 
                                               float(np.max(finite_data)))
        
        # Calculate valid count
        total_invalid = (invalid_info['nan_count'] + 
                        invalid_info['inf_count'] + 
                        invalid_info['out_of_range_count'])
        invalid_info['valid_count'] = invalid_info['total_values'] - total_invalid
        invalid_info['valid_percentage'] = (invalid_info['valid_count'] / 
                                           invalid_info['total_values'] * 100)
        
        results[var] = invalid_info
        
        # Print verbose output
        if verbose:
            print(f"\n{'='*60}")
            print(f"Variable: {var}")
            print(f"{'='*60}")
            print(f"Total values: {invalid_info['total_values']:,}")
            print(f"Valid values: {invalid_info['valid_count']:,} "
                  f"({invalid_info['valid_percentage']:.2f}%)")
            
            if invalid_info['nan_count'] > 0:
                print(f"NaN values: {invalid_info['nan_count']:,}")
            if invalid_info['inf_count'] > 0:
                print(f"Inf values: {invalid_info['inf_count']:,}")
            if invalid_info['out_of_range_count'] > 0:
                print(f"Out of range: {invalid_info['out_of_range_count']:,}")
                print(f"  Specified range: {invalid_info['specified_range']}")
                print(f"  Actual range: {invalid_info['actual_range']}")
    
    return results
#---------------------------------------------------------------------------------------------------
# Vertical Integration routines
#---------------------------------------------------------------------------------------------------
def calc_dp3d(ps,lev):
    """
    calculate pressure thickness for vertical integration
    both inputs should be xarray DataArray type
    ps  = surface pressure
    lev = pressure levels of the data (not the model coordinate)
    """

    ps3d,pl3d = xr.broadcast(ps,lev*100.)

    pl3d = pl3d.transpose('time','lev','ncol')
    ps3d = ps3d.transpose('time','lev','ncol')

    nlev = len(lev)
    tvals = slice(0,nlev-2)
    cvals = slice(1,nlev-1)
    bvals = slice(2,nlev-0)

    # Calculate pressure thickness
    dp3d = xr.DataArray( np.full(pl3d.shape,np.nan), coords=pl3d.coords )
    dp3d[:,nlev-1,:] = ps3d[:,nlev-1,:].values - pl3d[:,nlev-1,:].values
    dp3d[:,cvals,:] = pl3d[:,bvals,:].values - pl3d[:,tvals,:].values

    # Deal with cases where levels are below surface pressure
    condition = pl3d[:,cvals,:].values<ps3d[:,cvals,:].values
    new_data  = ps3d[:,bvals,:].values-pl3d[:,tvals,:].values
    dp3d[:,cvals,:] = dp3d[:,cvals,:].where( condition, new_data )

    # Screen out negative dp values
    dp3d = dp3d.where( dp3d>0, np.nan )

    return dp3d

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
