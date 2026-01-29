from hapy_common import *
#---------------------------------------------------------------------------------------------------
# Filtering routines
#---------------------------------------------------------------------------------------------------
def filter_wgts_lp_lanczos(window_len, fc_lp=0, fc_hp=0, sigma_factor=1):
   """
   Calculate weights for a low pass Lanczos filter taken from:
   Duchon C. E. (1979) Lanczos Filtering in One and Two Dimensions. 
      Journal of Applied Meteorology, Vol 18, pp 1016-1022.
   note: 0 < fc_lp < fc_lp < 0.5
   Args:
      window_len: The length of the filter window
      fc_lp: low-pass cutoff frequency in inverse time steps (set to zero to disable)
      fc_hp: high-pass cutoff frequency in inverse time steps (set to zero to disable)
   """
   order = ((window_len - 1) // 2 ) + 1
   nwts = 2 * order + 1
   w = np.zeros([nwts])
   n = nwts // 2
   k = np.arange(1., n)
   # Sigma factor to remove Gibbs oscillation
   sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
   sigma = np.power(sigma,sigma_factor)
   # specify filter weights
   factor_lp = np.sin(2. * np.pi * fc_lp * k) / (np.pi * k)
   factor_hp = np.sin(2. * np.pi * fc_hp * k) / (np.pi * k)
   w[n] = 2 * ( fc_lp - fc_hp )
   w[n-1:0:-1] = ( factor_lp - factor_hp ) * sigma
   w[n+1:-1]   = ( factor_lp - factor_hp ) * sigma
   return w[1:-1]