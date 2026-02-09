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
# misc methods that aren't easily classified
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
def print_time_length(time,indent='    ',print_msg=True,color=None,
                             print_span=True, print_length=True):
    """ print length of time """
    if len(time)<=1: return
    max_dy_for_mn = np.timedelta64(60, 'D')
    max_mn_for_yr = np.timedelta64(12, 'M')
    # calculate time delta
    dt = time[1] - time[0]
    # calculate length of period spanned by coordinate
    time_span_dy = ( time[-1] - time[0] + dt ).values.astype('timedelta64[D]') 
    time_span_mn = time_span_dy.astype('timedelta64[M]') + 1
    time_span_yr = time_span_mn.astype('timedelta64[Y]') + 1
    # calculate the actual length of the data (considering gaps with no data)
    time_leng_dy = (len(time) * dt).values.astype('timedelta64[D]') 
    time_leng_mn = time_leng_dy.astype('timedelta64[M]') + 1
    time_leng_yr = time_leng_mn.astype('timedelta64[Y]') + 1
    # build a message to be printed
    msg1 = indent+f'Time span   : {str(time_span_dy)}'
    msg2 = indent+f'Time length : {str(time_leng_dy)}'
    if time_span_dy > max_dy_for_mn : msg1 = msg1+f'  /  {time_span_mn}'
    if time_leng_dy > max_dy_for_mn : msg2 = msg2+f'  /  {time_leng_mn}'
    if time_span_mn > max_mn_for_yr : msg1 = msg1+f'  /  {time_span_yr}'
    if time_leng_mn > max_mn_for_yr : msg2 = msg2+f'  /  {time_leng_yr}'
    # add color if requested
    if color is not None:
        msg1 = f'{color}{msg1}{tcolor.ENDC}'
        msg2 = f'{color}{msg2}{tcolor.ENDC}'
    # print the formatted message
    if print_msg:
        if print_span:    print(msg1)
        if print_length:  print(msg2)
        return
    else:
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
