import os, subprocess as sp, xarray as xr, numpy as np, dask, numba
#---------------------------------------------------------------------------------------------------
# Binning Routines
#---------------------------------------------------------------------------------------------------
def bin_YbyX (Vy,Vx,bins=[],bin_min=0,bin_max=1,bin_spc=1,bin_spc_log=20,nbin_log=2,
              bin_mode="manual",verbose=False,wgt=None,
              keep_time=False,keep_lev=False,method='mean'):
   """ Average Vy into bins of Vx values according to bins and bin_mode. 
   Manual mode takes an array bin center values and determines thebin spacings. 
   Explicit mode takes a list of bin edge values, which is useful for an 
   irregularly spacing, such as logarithmic.  """
   #----------------------------------------------------------------------------
   # Manual mode - use min, max, and spc (i.e. stride) to define bins
   if bin_mode == "manual":
      # bins = np.arange(bin_min, bin_max+bin_spc, bin_spc)
      # nbin = len(bin_coord)
      nbin    = np.round( ( bin_max - bin_min + bin_spc )/bin_spc ).astype(int)
      bins    = np.linspace(bin_min,bin_max,nbin)
      bin_coord = xr.DataArray( bins )
   #----------------------------------------------------------------------------
   # Explicit mode - requires explicitly defined bin edges
   if bin_mode == "explicit":
      bins = xr.DataArray( bins )   # convert the input to a DataArray
      nbin = len(bins)-1
      bin_coord = ( bins[0:nbin-1+1] + bins[1:nbin+1] ) /2.
   #----------------------------------------------------------------------------
   # Log mode - logarithmically spaced bins that increase in width by bin_spc_log [%]
   if bin_mode == "log":
      bin_log_wid = np.zeros(nbin_log)
      bin_log_ctr = np.zeros(nbin_log)
      bin_log_ctr[0] = bin_min
      bin_log_wid[0] = bin_spc
      for b in range(1,nbin_log):
         bin_log_wid[b] = bin_log_wid[b-1] * (1.+bin_spc_log/1e2)  # note - bin_spc_log is in %
         bin_log_ctr[b] = bin_log_ctr[b-1] + bin_log_wid[b-1]/2. + bin_log_wid[b]/2.
      nbin = nbin_log
      bin_coord = xr.DataArray( bin_log_ctr )
      ### Print coords
      # for b in bin_coord.values : print( '{0:8.2f}'.format(b) )
   #----------------------------------------------------------------------------
   # create output data arrays
   nlev  = len(Vy['lev'])  if 'lev'  in Vy.dims else 1
   ntime = len(Vy['time']) if 'time' in Vy.dims else 1

   if ntime==1 and keep_time==True : keep_time = False

   shape,dims,coord = (nbin,),'bin',[('bin', bin_coord)]
   
   if nlev>1 and keep_lev and not keep_time :
      coord = [ ('bin', bin_coord.values), ('lev', Vy['lev'].values) ]
      shape = (nbin,nlev)
      dims = ['bin','lev']   
   if nlev==1 and not keep_lev and not keep_time :
      shape,dims,coord = (nbin,),'bin',[('bin', bin_coord.values)]
   # if nlev>1 and keep_time==True :
   #    coord = [ ('bin', bin_coord), ('time', Vy['time']), ('lev', Vy['lev']) ]
   #    shape = (nbin,ntime,nlev)
   #    dims = ['bin','time','lev']   
   if nlev==1 and keep_time==True :
      coord = [('bin', bin_coord.values), ('time', Vy['time'].values)]
      shape = (nbin,ntime)
      dims = ['bin','time']
   
   mval = np.nan
   bin_val = xr.DataArray( np.full(shape,mval,dtype=Vy.dtype), coords=coord, dims=dims )
   bin_std = xr.DataArray( np.full(shape,mval,dtype=Vy.dtype), coords=coord, dims=dims )
   bin_cnt = xr.DataArray( np.zeros(shape,dtype=Vy.dtype), coords=coord, dims=dims )
   #----------------------------------------------------------------------------
   # set up wgts
   # if wgt==[]: wgt, *__ = xr.broadcast(np.ones( Vy.shape[0] ), Vy)
   # if wgt is None: wgt = np.ones( Vy.shape )
   # if len(wgt)==0 : 
   #    wgt = np.ones( Vy.shape )
   # else:
   #    wgt = np.ma.masked_array( wgt, condition)

   lev_chk=False
   if 'lev' in Vy.dims and len(Vy.lev)>1 and keep_lev : lev_chk = True

   # if 'lev' in Vy.dims : 
   #    if len(Vy.lev)>1 : 
   #       lev_chk = True
   #    else:
   #       print("!!!!!!!!!")
   # else:
   #    lev_chk=False

   if lev_chk :
      avg_dims = ['ncol']
      if 'time' in Vy.dims : avg_dims = ['time','ncol']
      avg_dims_wgt = ['ncol']
      # if len(wgt)!=0 : 
         # if 'time' in wgt.dims : 
         #    avg_dims_wgt = ['time','ncol']
         # else:
         #    avg_dims_wgt = ['ncol']

   val_chk = np.isfinite(Vx.values)

   use_masked_arrays = False
   # if dask.is_dask_collection(Vy) and not lev_chk:
   #    # use masked array method to avoid numpy division warnings on dask arrays
   #    use_masked_arrays = True
   #    mask = np.logical_and( np.isfinite(Vy.compute().values), np.isfinite(Vx.compute().values) )
   #    Vy_tmp = Vy.load().data
   #    Vx_tmp = Vx.load().data
   #    Vy_tmp = np.ma.masked_array( np.where(Vy_tmp,Vy_tmp,-999), mask)
   #    Vx_tmp = np.ma.masked_array( np.where(Vx_tmp,Vx_tmp,-999), mask)

   if keep_time and 'time' in Vy.dims:
      if wgt.dims != Vy.dims : 
         wgt, *__ = xr.broadcast(wgt, Vy) 
         wgt = wgt.transpose('time','ncol')

   #----------------------------------------------------------------------------
   # Loop through bins
   for b in range(nbin):
      if bin_mode == "manual":
         bin_bot = bin_min - bin_spc/2. + bin_spc*(b  )
         bin_top = bin_min - bin_spc/2. + bin_spc*(b+1)
      if bin_mode == "explicit":
         bin_bot = bins[b]  .values
         bin_top = bins[b+1].values
      if bin_mode == "log":
         bin_bot = bin_log_ctr[b] - bin_log_wid[b]/2.
         bin_top = bin_log_ctr[b] + bin_log_wid[b]/2.

      condition = xr.DataArray( np.full(Vx.shape,False,dtype=bool), coords=Vx.coords )
      condition.values = ( np.where(val_chk,Vx.values,bin_bot-1e3) >=bin_bot ) \
                        &( np.where(val_chk,Vx.values,bin_bot-1e3)  <bin_top )

      # print(); print(condition)
      # tmp_data = Vx.where(condition,drop=True)
      # tmp_data = wgt.where(condition,drop=True)
      # print(); print(tmp_data.min().values)
      # print(); print(tmp_data.mean().values)
      # print(); print(tmp_data.max().values)
      # print(); print(tmp_data.sum().values)
      # exit()
      # print(f' bot: {bin_bot} cnt: {np.sum(condition.values)}')

      if np.sum(condition.values)>0 :
         if lev_chk :
            ### xarray method
            if len(wgt)==0 : 
               bin_val[b,:] = Vy.where(condition,drop=True).mean( dim=avg_dims, skipna=True )
            else:
               if wgt.dims != Vy.dims : 
                  wgt, *__ = xr.broadcast(wgt, Vy) 
                  if 'time' in Vy.dims :
                     wgt = wgt.transpose('time','lev','ncol')
                  else :
                     wgt = wgt.transpose('lev','ncol')
               if 'time' in Vy.dims : 
                  bin_val[b,:] = ( (Vy*wgt).where(condition,drop=True).sum( dim='ncol', skipna=True ) \
                                      / wgt.where(condition,drop=True).sum( dim='ncol', skipna=True ) ).mean(dim='time', skipna=True )
               else:
                  bin_val[b,:] = ( (Vy*wgt).where(condition,drop=True).sum( dim='ncol', skipna=True ) \
                                      / wgt.where(condition,drop=True).sum( dim='ncol', skipna=True ) )
            bin_std[b,:] = Vy.where(condition,drop=True).std(  dim=avg_dims, skipna=True )
            bin_cnt[b,:] = Vy.where(condition,drop=True).count(dim=avg_dims)

            ### masked array method - Not sure how to make this work with a lev dimension...
            # bin_val[b,:] = np.sum( np.where(condition, Vy_tmp*wgt, 0 ) ) / np.sum( np.where(condition, wgt, 0 ) )
            # bin_val[b,:] = bin_val[b,:] / wgt.where(condition,drop=True).sum( dim=['time','ncol'])
            # bin_std[b,:] = np.where(condition, np.std( Vy_tmp ) )
            # bin_cnt[b,:] = np.sum( condition )
         elif keep_time and 'time' in Vy.dims:
            # avg_dims = ['ncol']
            # if wgt.dims != Vy.dims : 
            #    wgt, *__ = xr.broadcast(wgt, Vy) 
            #    wgt = wgt.transpose('time','ncol')
            bin_val[b,:] = ( (Vy*wgt).where(condition,drop=True).sum( dim='ncol', skipna=True ) \
                                / wgt.where(condition,drop=True).sum( dim='ncol', skipna=True ) )
            bin_cnt[b] = np.sum( np.where(condition, 1, 0 ) )
         else:
            if use_masked_arrays:
               ### numpy masked array method
               with np.errstate(divide='ignore', invalid="ignore"):
                  print("!")
                  # tmp = np.sum( np.where(condition, Vy_tmp*wgt, 0 ) )
                  # print("!")
                  # tmp = np.sum( np.where(condition, wgt, 0 ) )
                  bin_cnt[b] = np.sum( np.where(condition, 1, 0 ) )
                  exit()

               if method=='max':
                  bin_val[b] = Vy_tmp.max(skipna=True)
               if method=='std':
                  bin_val[b] = Vy_tmp.std(skipna=True)
               if method=='mean':
                  bin_val[b] = np.sum( np.where(condition, Vy_tmp*wgt, 0 ) ) / np.sum( np.where(condition, wgt, 0 ) )
               bin_std[b] = np.where(condition, np.std( Vy_tmp ) )
               bin_cnt[b] = np.sum( condition )
            else:
               ### xarray method - can't avoid divide warnings with dask arrays
               Vy.load()
               # print(); print(Vy)
               # print(); print(Vy*wgt)
               # exit()
               if method=='max':
                  bin_val[b] = Vy.where(condition).max(skipna=True)
               if method=='std':
                  bin_val[b] = Vy.where(condition).std(skipna=True)
               if method=='mean':
                  if wgt is None:
                     bin_val[b] = Vy.where(condition).mean(skipna=True)
                  else:
                     bin_val[b] = (Vy*wgt).where(condition).sum(skipna=True) / wgt.where(condition).sum(skipna=True)
               # bin_val[b] = Vy.where(condition).mean(skipna=True)
               bin_std[b] = Vy.where(condition).std(skipna=True)
               bin_cnt[b] = np.sum( condition )
            
      # print('b: '+str(b)+'  cnt: '+str(bin_cnt[b])+'  val: '+str(bin_val[b]))
      if not lev_chk and verbose : print('b: {0:8.0f}  cnt: {1:8.0f}  val: {2:12.4e} '.format( b, bin_cnt[b].values, bin_val[b].values ) )
   #----------------------------------------------------------------------------
   ### add mask (doesn't propagate to xarray?)
   # bin_val = np.ma.masked_invalid(bin_val)
   # bin_std = np.ma.masked_invalid(bin_std)
   #----------------------------------------------------------------------------
   
   # use a dataset to hold all the output
   dims = ('bins',)
   if lev_chk and not keep_time : dims = ('bins','lev')
   if not lev_chk and keep_time : dims = ('bins','time')
   if lev_chk and keep_time     : dims = ('bins','time','lev')

   bin_ds = xr.Dataset()
   bin_ds['bin_val'] = bin_val
   bin_ds['bin_std'] = bin_std
   bin_ds['bin_cnt'] = bin_cnt
   bin_ds['bin_pct'] = bin_cnt/bin_cnt.sum()*1e2
   bin_ds.coords['bins'] = bin_coord
   if lev_chk : bin_ds.coords['lev'] = xr.DataArray(Vy['lev'])

   #----------------------------------------------------------------------------
   return bin_ds
#-------------------------------------------------------------------------------
def bin_ZbyYX (Vz,Vy,Vx,binsy,binsx,opt):
   """ """
#-------------------------------------------------------------------------------

