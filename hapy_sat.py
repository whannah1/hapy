from hapy_common import *
#---------------------------------------------------------------------------------------------------
# Saturation routines
#---------------------------------------------------------------------------------------------------
# Calculate saturation specific humidity (kg/kg) from temperature (k) and pressure (mb)
# Buck Research Manual (1996)
def calc_qsat (Ta,Pa) :
   Tc = Ta - 273.
   ew = 6.1121*(1.0007+3.46e-6*Pa)*np.exp((17.502*Tc)/(240.97+Tc))       # in mb
   qs = 0.62197*(ew/(Pa-0.378*ew))                                       # mb -> kg/kg
   return qs
#---------------------------------------------------------------------------------------------------
# Calculate saturation specific humidity (kg/kg) from temperature (k) and pressure (mb)
def calc_Qsat_Peters (Ta,Pa):
   qs = (380./Pa)*np.exp(17.27*(Ta-273.0)/(Ta-36.0))
   return qs
#---------------------------------------------------------------------------------------------------