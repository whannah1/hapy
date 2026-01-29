from hapy_common import *
#---------------------------------------------------------------------------------------------------
# Spherical Calculation routines
#---------------------------------------------------------------------------------------------------
# calculate radius of Earth assuming oblate spheroid
def earth_radius(lat):
   # lat: vector or latitudes in degrees  
   # r: vector of radius in meters
   # WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
   from numpy import deg2rad, sin, cos

   # define oblate spheroid from WGS84
   a = 6378137
   b = 6356752.3142
   e2 = 1 - (b**2/a**2)

   # convert from geodecic to geocentric
   # see equation 3-110 in WGS84
   lat = deg2rad(lat)
   lat_gc = np.arctan( (1-e2)*np.tan(lat) )

   # radius equation
   # see equation 3-107 in WGS84
   r = ( (a * (1 - e2)**0.5) / (1 - (e2 * np.cos(lat_gc)**2))**0.5 )

   return r
#---------------------------------------------------------------------------------------------------
# find nearest neighbors
@numba.njit()
def find_nearest_neighbors_on_sphere_numba(lat,lon,num_neighbors,neighbor_id) :
   # Note - input should be radians
   ncol = len(lat)
   for n in range(0,ncol) :
      cos_dist = np.sin(lat[n])*np.sin(lat[:]) + \
                 np.cos(lat[n])*np.cos(lat[:]) * np.cos( lon[n]-lon[:] )
      for nn in range(0,ncol+1) :
         if cos_dist[nn] >  1.0 : cos_dist[nn] = 1.0
         if cos_dist[nn] < -1.0 : cos_dist[nn] = -1.0
      dist = np.arccos( cos_dist )
      p_vector = np.argsort(dist,kind='mergesort')
      neighbor_id[n,0:num_neighbors] = p_vector[1:num_neighbors+1]
#---------------------------------------------------------------------------------------------------
def find_nearest_neighbors_on_sphere(lat,lon,num_neighbors,return_sorted=False) :
   # Note - input should be degrees
   # return_sorted = False
   ncol = len(lat)
   neighbor_id    = np.empty([num_neighbors+1,ncol], dtype=np.int32)
   neighbor_dist  = np.empty([num_neighbors+1,ncol], dtype=np.float32)
   for n in range(0,ncol):
      dist = calc_great_circle_distance(lat[n],lat[:],lon[n],lon[:])
      if return_sorted:
         p_vector = np.argsort(dist,kind='mergesort')
         neighbor_id[:,n]   = p_vector[0:num_neighbors+1]     # include current point
         neighbor_dist[:,n] = dist[p_vector[0:num_neighbors+1]]
      else:
         p_vector = np.argpartition(dist[1:],num_neighbors)+1
         neighbor_id[:,n] = p_vector[0:num_neighbors]
   # return neighbor_id
   neighbor_ds = xr.Dataset()
   neighbor_ds['neighbor_id']   = (('ncol','neighbors'), neighbor_id)
   neighbor_ds['neighbor_dist'] = (('ncol','neighbors'), neighbor_dist)
   return neighbor_ds
#---------------------------------------------------------------------------------------------------
# input should be in degrees
def calc_great_circle_distance(lat1,lat2,lon1,lon2):
   dlon = lon2 - lon1
   cos_dist = np.sin(lat1*deg_to_rad)*np.sin(lat2*deg_to_rad) + \
              np.cos(lat1*deg_to_rad)*np.cos(lat2*deg_to_rad)*np.cos(dlon*deg_to_rad)
   # print( str(cos_dist.min()) +"   "+ str(cos_dist.max()) )
   cos_dist = np.where(cos_dist> 1.0, 1.0,cos_dist)
   cos_dist = np.where(cos_dist<-1.0,-1.0,cos_dist)
   dist = np.arccos( cos_dist )
   return dist
#---------------------------------------------------------------------------------------------------
# @numba.njit
# def calc_great_circle_bearing(lat1_in,lat2_in,lon1_in,lon2_in):
#    lat1 = lat1_in * deg_to_rad
#    lat2 = lat2_in * deg_to_rad
#    lon1 = lon1_in * deg_to_rad
#    lon2 = lon2_in * deg_to_rad

#    dlon = lon1 - lon2

#    atan_tmp1 = sin(lon2-lon1)*cos(lat2)
#    atan_tmp2 = cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1)
#    bearing = atan2( atan_tmp1, atan_tmp2 )

#    bearing = bearing * rad_to_deg
#    return bearing

#---------------------------------------------------------------------------------------------------
# def calc_sphereical_triangle_area (lat1,lat2,lat3,lon1,lon2,lon3):
#     ;;; compute great circle lengths
#     al = calc_great_circle_distance(lat2, lat3, lon2, lon3 ) 
#     bl = calc_great_circle_distance(lat1, lat3, lon1, lon3 ) 
#     cl = calc_great_circle_distance(lat1, lat2, lon1, lon2 ) 

#     ;;; compute angles
#     sina = sin( ( bl + cl - al ) /2. )
#     sinb = sin( ( al + cl - bl ) /2. )
#     sinc = sin( ( al + bl - cl ) /2. ) 
#     sins = sin( ( al + bl + cl ) /2. )

#     a = sqrt( (sinb*sinc) / (sins*sina) ) 
#     b = sqrt( (sina*sinc) / (sins*sinb) ) 
#     c = sqrt( (sina*sinb) / (sins*sinc) ) 

#     a1 = 2.*atan(a)
#     b1 = 2.*atan(b)
#     c1 = 2.*atan(c)

#     if ( a.gt.b .and. a.gt.c ) then
#         ; a1 = -2*atan(1/a)
#         a1 = -2.*atan2(1.,a)
#     else 
#         if (b.gt.c) then
#             ; b1 = -2.*atan(1./b)
#             b1 = -2.*atan2(1.,b)
#         else 
#             ; c1 = -2.*atan(1./c)
#             c1 = -2.*atan2(1.,c)
#         end if
#     end if

#     # apply Girard's theorem
#     area = totype( a1+b1+c1 ,typeof(lat1))
#     return area
#---------------------------------------------------------------------------------------------------

