import os
import subprocess as sp
import xarray as xr
import numpy as np
import dask
import numba
from scipy.interpolate import interp1d
#---------------------------------------------------------------------------------------------------
# terminal color class
class tclr:
    END          = '\033[0m'
    BLACK        = '\033[30m'
    RED          = '\033[31m'
    GREEN        = '\033[32m'
    YELLOW       = '\033[33m' 
    BLUE         = '\033[34m'
    MAGENTA      = '\033[35m'
    CYAN         = '\033[36m'
    WHITE        = '\033[37m'
    DGRAY        = '\033[90m'
    LGRAY        = '\033[37m'
    LRED         = '\033[91m'
    LGREEN       = '\033[92m'
    LYELLOW      = '\033[93m'
    LBLUE        = '\033[94m'
    LMAGENTA     = '\033[95m'
    LCYAN        = '\033[96m'
    WHITE        = '\033[97m'
    BOLD         = '\033[1m'
    UNDERLINE    = '\033[4m'
#---------------------------------------------------------------------------------------------------
def print_line(n=80):
    """ print a line of n characters """
    print('-'*n)
#---------------------------------------------------------------------------------------------------