import numpy as np
#---------------------------------------------------------------------------------------------------
# Constants
pi      = np.pi
cpd     = 1004.         # J / (kg K)       heat capavity at constant pressure
cpv     = 1850.         # J / (kg K)       heat capavity at constant volume
Tf      = 273.15        # K                freezing temperature of water
g       = 9.81          # m / s^2          acceleration due to gravity
Lv      = 2.5104e6      # J / kg           latent heat of vaporization / evaporation
Lf      = 0.3336e6      # J / kg           latent heat of freezing / melting
Ls      = 2.8440e6      # J / kg           latent heat of sublimation
Po      = 100000.       # Pa               reference pressure
Rd      = 287.04        # J / (kg K)       gas constant for dry air
Rv      = 461.5         # J / (kg K)       gas constant for water vapor
esf     = 611.          # Pa               ?
eps     = 0.622         # (epsilon)        ?

days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]

deg_to_rad = pi/180.
rad_to_deg = 180./pi
#---------------------------------------------------------------------------------------------------