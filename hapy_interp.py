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