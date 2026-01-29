from hapy_common import *
#---------------------------------------------------------------------------------------------------
def print_stat(x,name='(no name)',unit='',fmt='f',stat='naxh',indent=' '*2,compact=True):
   """ Print min, avg, max, and std deviation of input """
   if fmt=='f' : fmt = '%.4f'
   if fmt=='e' : fmt = '%e'
   if fmt=='g' : fmt = '%g'
   if unit!='' : unit = f'[{unit}]'
   name_len = 12 if compact else len(name)
   msg = ''
   line = f'{indent}{name:{name_len}} {unit}'
   # if not compact: print(line)
   if not compact: msg += line+'\n'
   for c in list(stat):
      if not compact: line = indent
      if c=='h' : line += '   shp: '+str(x.shape)
      if c=='a' : line += '   avg: '+fmt%x.mean()
      if c=='n' : line += '   min: '+fmt%x.min()
      if c=='x' : line += '   max: '+fmt%x.max()
      if c=='s' : line += '   std: '+fmt%x.std()
      # if not compact: print(line)
      if not compact: msg += line+'\n'
   # if compact: print(line)
   if compact: msg += line#+'\n'
   print(msg)
   return msg
#---------------------------------------------------------------------------------------------------
def median(array, dim, keep_attrs=False, skipna=False, **kwargs):
   """ Runs a median on an dask-backed xarray.

   This function does not scale!
   It will rechunk along the given dimension, so make sure 
   your other chunk sizes are small enough that it 
   will fit into memory.

   :param DataArray array: An xarray.DataArray wrapping a dask array
   :param dim str: The name of the dim in array to calculate the median
   """
   import dask.array
   if type(array) is xr.Dataset: return array.apply(median, dim=dim, keep_attrs=keep_attrs, **kwargs)

   if not hasattr(array.data, 'dask'): return array.median(dim, keep_attrs=keep_attrs, **kwargs)

   array = array.chunk({dim:-1})
   axis = array.dims.index(dim)
   median_func = np.nanmedian if skipna else np.median
   blocks = dask.array.map_blocks(median_func, array.data, dtype=array.dtype, drop_axis=axis, axis=axis, **kwargs)

   new_coords={k: v for k, v in array.coords.items() if k != dim and dim not in v.dims}
   new_dims = tuple(d for d in array.dims if d != dim)
   new_attrs = array.attrs if keep_attrs else None

   return xr.DataArray(blocks, coords=new_coords, dims=new_dims, attrs=new_attrs)
#---------------------------------------------------------------------------------------------------
def calc_t_stat( D0, D1, S0, S1, N0, N1, t_crit=1.96, verbose=True ):
   """ 
   calculate Student's t-statistic 
   t_crit = 1.96   2-tail test w/ inf dof & P=0.05
   t_crit = 2.5    2-tail test w/ 5 dof & P=0.05
   t_crit = 2.2    2-tail test w/ 10 dof & P=0.05
   """

   ### Standard error
   SE = np.sqrt( S0**2/N0 + S1**2/N1 )

   ### t-statistic - aX is the difference now
   t_stat = ( D1 - D0 ) / SE

   ### Degrees of freedom
   # DOF = (S0**2/N0 + S1**2/N1)**2 /( ( (S0**2/N0)**2 / (N0-1) )   \
   #                                  +( (S1**2/N1)**2 / (N1-1) ) )
   # print(f'  DoF min/max: {DOF.min():6.1f} / {DOF.max():6.1f}')

   # hc.print_stat(SE,name='SE',indent='    ')
   # hc.print_stat(t_stat,name='t statistic',indent='    ')

   ### Critical t-statistic
   

   sig_cnt = np.sum( np.absolute(t_stat) > t_crit )
   sig_pct = sig_cnt / t_stat.size *100

   if verbose: print(f'   SIG COUNT: {sig_cnt:8}   ({sig_pct:5.2f}% of {t_stat.size})')
   
   # for i in range(len(lat_bins)):
   #    msg = f'  lat: {lat_bins[i]}   t_stat: {t_stat[i]}   '
   #    if np.absolute(t_stat[i])>t_crit: msg = msg+tcolor.RED+'SIGNIFICANT'+tcolor.ENDC
   #    print(msg)

   return t_stat

#---------------------------------------------------------------------------------------------------
