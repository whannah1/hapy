from hapy_common import *
from scipy.signal import detrend
#---------------------------------------------------------------------------------------------------
# Wave analysis routines
#   Wheeler-Kiladis space-time spectra for data on unstructured (native) grids
#   the zonal FFT is replaced by an area-weighted projection onto zonal Fourier modes within
#   latitude bins that are symmetric about the equator - the rest of the calculation follows
#   wavenumber_frequency_functions.spacetime_power() (https://github.com/brianpm/wavenumber_frequency)
#   since detrending, tapering, and the symmetric/antisymmetric decomposition are all linear
#---------------------------------------------------------------------------------------------------
def _get_wf():
    """ import wavenumber_frequency_functions only when needed so it's not a hard dependency """
    try:
        import wavenumber_frequency_functions as wf
    except ImportError as e:
        raise ImportError('hapy_waves requires wavenumber_frequency_functions to be in PYTHONPATH '
                          '(https://github.com/brianpm/wavenumber_frequency)') from e
    return wf
#---------------------------------------------------------------------------------------------------
def _detrend_complex(x,axis):
    """ linear detrend of real and imaginary parts separately """
    return detrend(x.real,axis=axis,type='linear') + 1j*detrend(x.imag,axis=axis,type='linear')
#---------------------------------------------------------------------------------------------------
def zonal_fourier_native(data,lat,lon,area,lat2,dlat,kmax,time_chunk=365,verbose=True):
    """ project unstructured (time,ncol) data onto zonal Fourier modes within latitude bins
        X_k(lat_bin,t) = sum_j( A_j * x_j(t) * exp(-i*k*lon_j) ) / sum_j( A_j )
    data       : DataArray with (time,ncol) dimensions
    lat,lon    : column latitude/longitude [deg]
    area       : column area (any units)
    lat2       : bins cover -lat2 to lat2
    dlat       : bin width - must be larger than the grid spacing to avoid empty bins
    kmax       : max zonal wavenumber - should be well above the max wavenumber of interest
    time_chunk : number of time samples loaded at once
    bins are centered on multiples of dlat so they are symmetric about the equator
    normalization matches np.fft.fft(x,axis=lon)/nlon on a regular grid
    returns complex DataArray with (time,lat,wavenumber) dimensions,
    with wavenumber in standard FFT order (i.e. np.fft.fftfreq) """
    nb = int(np.round(lat2/dlat))
    lat_c = dlat*np.arange(-nb,nb+1)
    nbin = len(lat_c)
    lat_v,lon_v,area_v = [ np.asarray(x,dtype=np.float64) for x in (lat,lon,area) ]
    #----------------------------------------------------------------------------
    # find columns in the band and sort them by latitude bin
    lat_edge0 = lat_c[0]-dlat/2
    idx = np.where( (lat_v>=lat_edge0) & (lat_v<lat_c[-1]+dlat/2) )[0]
    bin_id = np.clip( np.floor((lat_v[idx]-lat_edge0)/dlat).astype(int), 0, nbin-1 )
    srt = np.argsort(bin_id, kind='stable')
    idx,bin_id = idx[srt],bin_id[srt]
    bin_cnt = np.bincount(bin_id, minlength=nbin)
    bin_end = np.cumsum(bin_cnt)
    bin_beg = bin_end - bin_cnt
    if np.any(bin_cnt==0): raise ValueError('empty latitude bin - increase dlat')
    if verbose: print(' '*6+f'columns per lat bin: min {bin_cnt.min()}  max {bin_cnt.max()}  (band total: {len(idx)})')
    if np.any(bin_cnt<2*kmax):
        print(' '*6+f'{tclr.RED}WARNING - some lat bins have fewer than 2*kmax={2*kmax} columns - expect aliasing{tclr.END}')
    #----------------------------------------------------------------------------
    # area-weighted Fourier projection matrix
    nlon = 2*kmax
    wvn = np.fft.fftfreq(nlon,1/nlon)
    wgt = area_v[idx]
    for b in range(nbin): wgt[bin_beg[b]:bin_end[b]] /= wgt[bin_beg[b]:bin_end[b]].sum()
    proj = wgt[:,None] * np.exp( -1j * np.outer(np.deg2rad(lon_v[idx]), wvn) )
    #----------------------------------------------------------------------------
    data_band = data.isel(ncol=idx)
    ntime = data_band.sizes['time']
    X = np.empty((ntime,nbin,nlon), dtype=np.complex128)
    for t1 in range(0,ntime,time_chunk):
        t2 = min(t1+time_chunk,ntime)
        vals = data_band.isel(time=slice(t1,t2)).values.astype(np.float64)
        for b in range(nbin):
            X[t1:t2,b,:] = vals[:,bin_beg[b]:bin_end[b]] @ proj[bin_beg[b]:bin_end[b],:]
    return xr.DataArray(X, dims=('time','lat','wavenumber'),
                        coords={'time':data_band['time'],'lat':lat_c,'wavenumber':wvn})
#---------------------------------------------------------------------------------------------------
def spacetime_power_from_zonal_fourier(X,segsize=96,noverlap=60,spd=1):
    """ same as wf.spacetime_power(..., dosymmetries=True) except that the input is the
        complex zonal Fourier coefficients (time,lat,wavenumber) rather than (time,lat,lon)
    X        : output of zonal_fourier_native()
    segsize  : segment length [days]
    noverlap : segment overlap [days]
    spd      : samples per day
    returns power DataArray with (component,wavenumber,frequency) dimensions
    where component = ['symmetric','antisymmetric'] """
    wf = _get_wf()
    segsize,noverlap = spd*segsize,spd*noverlap
    if segsize-noverlap<=0: raise ValueError('segsize must be larger than noverlap')
    lat = X['lat'].values
    xv = X.values
    if xv.shape[0]<segsize: raise ValueError(f'not enough time samples ({xv.shape[0]}) for segsize={segsize}')
    #----------------------------------------------------------------------------
    # detrend the full series (keep the mean, as in NCL)
    xv = _detrend_complex(xv,axis=0) + xv.mean(axis=0)
    #----------------------------------------------------------------------------
    # symmetric component in SH, antisymmetric in NH (equator is left alone)
    xflp = xv[:,::-1,:]
    out = xv.copy()
    out[:,lat<0,:] = ( 0.5*(xv+xflp) )[:,lat<0,:]
    out[:,lat>0,:] = ( 0.5*(xv-xflp) )[:,lat>0,:]
    #----------------------------------------------------------------------------
    # overlapping segments => (segment,lat,wavenumber,time)
    starts = range(0, out.shape[0]-segsize+1, segsize-noverlap)
    x_win = np.stack([ np.moveaxis(out[s:s+segsize],0,-1) for s in starts ])
    x_win = _detrend_complex(x_win,axis=-1)
    x_win = x_win * wf.split_hann_taper(segsize,0.1)
    #----------------------------------------------------------------------------
    z = np.fft.fft(x_win,axis=-1) / segsize
    z = xr.DataArray(z, dims=('time','lat','wavenumber','frequency'),
                     coords={'time':np.arange(len(starts)),'lat':lat,
                             'wavenumber':X['wavenumber'].values,
                             'frequency':np.fft.fftfreq(segsize,1/spd)})
    z_pee = wf.resolveWavesHayashi(z, segsize//spd, spd)
    #----------------------------------------------------------------------------
    # average over segments and sum over latitude (x2 b/c only one hemisphere is used)
    z_sym = ( 2.0*z_pee.isel(lat=z_pee.lat<0).mean(dim='time') ).sum(dim='lat')
    z_asy = ( 2.0*z_pee.isel(lat=z_pee.lat>0).mean(dim='time') ).sum(dim='lat')
    spec = xr.concat([z_sym,z_asy],'component')
    spec = spec.assign_coords({'component':['symmetric','antisymmetric']})
    spec.name = 'power'
    return spec
#---------------------------------------------------------------------------------------------------
def spacetime_power_native(data,lat,lon,area,lat2=15,dlat=2.5,kmax=30,
                           segsize=96,noverlap=60,spd=1,verbose=True):
    """ Wheeler-Kiladis space-time power spectrum for unstructured (time,ncol) data
        convenience wrapper for zonal_fourier_native() + spacetime_power_from_zonal_fourier() """
    X = zonal_fourier_native(data,lat,lon,area,lat2=lat2,dlat=dlat,kmax=kmax,verbose=verbose)
    return spacetime_power_from_zonal_fourier(X,segsize=segsize,noverlap=noverlap,spd=spd)
#---------------------------------------------------------------------------------------------------
