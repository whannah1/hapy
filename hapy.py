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
