from hapy_common import *
import hapy_constants as constants
#---------------------------------------------------------------------------------------------------
# hapy_cape.py
#
# Undilute (pseudoadiabatic) Convective Available Potential Energy (CAPE).
#
# A single surface-based parcel is lifted dry-adiabatically to its lifting
# condensation level and pseudoadiabatically above it.  The buoyancy of the
# parcel relative to its environment is integrated in log-pressure to yield
# CAPE (and, optionally, CIN).  "Undilute" means the ascending parcel does not
# entrain/mix with environmental air.
#
# The calculation only needs temperature, humidity, and pressure at each level,
# so the same core routine handles data on pressure levels, height levels, or
# E3SM hybrid sigma-pressure levels -- the only difference is how the pressure
# field is obtained (see calc_cape()).  Any leading dimensions (e.g. time and
# the horizontal/column dimensions) are preserved; only the vertical dimension
# is collapsed.
#---------------------------------------------------------------------------------------------------
# Thermodynamic constants (plain floats so numba can capture them at compile)
Rd   = float(constants.Rd)    # J/(kg K)  gas constant for dry air
Rv   = float(constants.Rv)    # J/(kg K)  gas constant for water vapor
cpd  = float(constants.cpd)   # J/(kg K)  specific heat of dry air at constant pressure
Lv   = float(constants.Lv)    # J/kg      latent heat of vaporization
eps  = float(constants.eps)   #           Rd/Rv ~ 0.622
#---------------------------------------------------------------------------------------------------
# candidate names used to auto-detect the vertical dimension
_vert_dim_candidates = ['lev','plev','pressure_level','pressure','ilev','height','z','model_level']
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _sat_vapor_pressure(T):
    """ Saturation vapor pressure over liquid water [Pa], Bolton (1980). """
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _sat_mixing_ratio(T, p):
    """ Saturation mixing ratio [kg/kg] at temperature T [K] and pressure p [Pa]. """
    es = _sat_vapor_pressure(T)
    # keep es below p to avoid a singular/negative denominator in thin cold air
    if es > 0.5 * p: es = 0.5 * p
    return eps * es / (p - es)
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _dT_dlnp(T, p, rp, saturated):
    """ Lapse rate dT/d(ln p) [K] for a lifted parcel.
    Dry adiabat when unsaturated; pseudoadiabat (with saturation mixing ratio
    rs = rs(T,p)) once the parcel is saturated. """
    if not saturated:
        return Rd * T / cpd
    rs = _sat_mixing_ratio(T, p)
    return (Rd * T + Lv * rs) / (cpd + Lv * Lv * rs / (Rv * T * T))
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _virtual_temperature(T, r):
    """ Virtual temperature [K] from temperature T [K] and mixing ratio r [kg/kg]. """
    return T * (1.0 + r / eps) / (1.0 + r)
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _lift_parcel(T0, p0, r0, p_target, saturated, r_parcel, n_sub):
    """ Integrate a lifted parcel from pressure p0 to p_target in log-pressure.
    Returns (T_parcel, r_parcel, saturated) at p_target.  Uses a midpoint (RK2)
    step with n_sub substeps and switches from the dry to the pseudoadiabat the
    first time the parcel becomes saturated. """
    T = T0
    rp = r_parcel
    lnp0 = np.log(p0)
    lnpt = np.log(p_target)
    dlnp = (lnpt - lnp0) / n_sub
    lnp  = lnp0
    for _ in range(n_sub):
        # ------------------------------------------------------------
        # detect the onset of saturation before taking the step
        # ------------------------------------------------------------
        if not saturated:
            if _sat_mixing_ratio(T, np.exp(lnp)) <= r0:
                saturated = True
        # ------------------------------------------------------------
        # midpoint (RK2) integration of the lapse rate
        # ------------------------------------------------------------
        k1    = _dT_dlnp(T, np.exp(lnp), rp, saturated)
        Tmid  = T + 0.5 * dlnp * k1
        lnmid = lnp + 0.5 * dlnp
        k2    = _dT_dlnp(Tmid, np.exp(lnmid), rp, saturated)
        T     = T + dlnp * k2
        lnp   = lnp + dlnp
        # ------------------------------------------------------------
        # keep the saturated parcel on its saturation mixing ratio
        # ------------------------------------------------------------
        if saturated:
            rp = _sat_mixing_ratio(T, np.exp(lnp))
        else:
            rp = r0
    return T, rp, saturated
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _cape_cin_column(p, T, r, virtual, n_sub):
    """ Undilute CAPE and CIN [J/kg] for a single column.
    Inputs are 1-D arrays over the vertical dimension (any order, NaNs allowed).
    A parcel is launched from the lowest valid (highest pressure) level. """
    nlev = p.shape[0]
    # --------------------------------------------------------------------
    # gather valid levels and sort them from the surface (high p) upward
    # --------------------------------------------------------------------
    nv = 0
    for k in range(nlev):
        if np.isfinite(p[k]) and np.isfinite(T[k]) and np.isfinite(r[k]):
            nv += 1
    if nv < 2:
        return np.nan, np.nan
    pv = np.empty(nv); Tv = np.empty(nv); rv = np.empty(nv)
    j = 0
    for k in range(nlev):
        if np.isfinite(p[k]) and np.isfinite(T[k]) and np.isfinite(r[k]):
            pv[j] = p[k]; Tv[j] = T[k]; rv[j] = r[k]; j += 1
    order = np.argsort(pv)[::-1]   # descending pressure -> surface first
    pv = pv[order]; Tv = Tv[order]; rv = rv[order]
    # --------------------------------------------------------------------
    # initialize the surface parcel
    # --------------------------------------------------------------------
    p_sfc = pv[0]; T_sfc = Tv[0]; r_sfc = rv[0]
    saturated = _sat_mixing_ratio(T_sfc, p_sfc) <= r_sfc
    Tp_prev = T_sfc
    rp_prev = r_sfc if not saturated else _sat_mixing_ratio(T_sfc, p_sfc)
    # buoyancy at the launch level is zero by construction
    if virtual:
        B_prev = _virtual_temperature(Tp_prev, rp_prev) - _virtual_temperature(T_sfc, r_sfc)
    else:
        B_prev = Tp_prev - T_sfc
    # --------------------------------------------------------------------
    # march upward, integrating buoyancy in log-pressure
    # --------------------------------------------------------------------
    cape = 0.0
    cin  = 0.0
    found_lfc = False
    for k in range(1, nv):
        p_lev = pv[k]
        Tp, rp, saturated = _lift_parcel(Tp_prev, pv[k-1], r_sfc, p_lev,
                                         saturated, rp_prev, n_sub)
        if virtual:
            B = _virtual_temperature(Tp, rp) - _virtual_temperature(Tv[k], rv[k])
        else:
            B = Tp - Tv[k]
        # trapezoidal layer contribution: -Rd * B * d(ln p) > 0 when buoyant
        dcontrib = -Rd * 0.5 * (B_prev + B) * (np.log(p_lev) - np.log(pv[k-1]))
        if dcontrib > 0.0:
            cape += dcontrib
            found_lfc = True
        elif not found_lfc:
            # negative area below the level of free convection is inhibition
            cin += dcontrib
        Tp_prev = Tp; rp_prev = rp; B_prev = B
    return cape, cin
#---------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _cape_cin_field(p2d, T2d, r2d, virtual, n_sub):
    """ Loop _cape_cin_column() over the leading (flattened column) axis. """
    ncol = p2d.shape[0]
    cape = np.empty(ncol)
    cin  = np.empty(ncol)
    for i in range(ncol):
        cape[i], cin[i] = _cape_cin_column(p2d[i], T2d[i], r2d[i], virtual, n_sub)
    return cape, cin
#---------------------------------------------------------------------------------------------------
def _to_pascals(p):
    """ Return pressure in Pa, converting from hPa if the magnitudes look like hPa. """
    pmax = float(np.nanmax(np.asarray(p if not isinstance(p, xr.DataArray) else p.values)))
    if pmax < 2000.0:   # ~ surface pressure in hPa is ~1013, in Pa is ~101300
        return p * 100.0
    return p
#---------------------------------------------------------------------------------------------------
def _find_vert_dim(da, lev_dim):
    """ Resolve the vertical dimension name, auto-detecting when not given. """
    if lev_dim is not None:
        if lev_dim not in da.dims:
            raise ValueError(f"lev_dim '{lev_dim}' not found in dims {list(da.dims)}")
        return lev_dim
    found = [d for d in _vert_dim_candidates if d in da.dims]
    if not found:
        raise ValueError(f"Could not auto-detect a vertical dimension in {list(da.dims)}. "
                         f"Pass lev_dim explicitly.")
    return found[0]
#---------------------------------------------------------------------------------------------------
def _build_pressure(T, lev_dim, pressure, ps, hyam, hybm, p0):
    """ Construct a pressure field [Pa] broadcast to the shape of T for any of the
    supported vertical coordinates (explicit pressure, hybrid, or a pressure-like
    level coordinate). """
    #----------------------------------------------------------------------------
    # 1. explicit pressure supplied (pressure levels, or height levels with p)
    #----------------------------------------------------------------------------
    if pressure is not None:
        p = pressure
        if not isinstance(p, xr.DataArray):
            p = np.asarray(p)
            if p.ndim == 1:
                p = xr.DataArray(p, dims=[lev_dim])
            else:
                raise ValueError("Multi-dimensional 'pressure' must be an xarray.DataArray "
                                 "so its dimensions can be aligned with the temperature.")
    #----------------------------------------------------------------------------
    # 2. E3SM hybrid sigma-pressure levels: p = hyam*p0 + hybm*ps
    #----------------------------------------------------------------------------
    elif hyam is not None and hybm is not None and ps is not None:
        if p0 is None: p0 = 1.0e5
        ps = _to_pascals(ps)
        p  = hyam * p0 + hybm * ps
    #----------------------------------------------------------------------------
    # 3. fall back to a pressure-like vertical coordinate on T
    #----------------------------------------------------------------------------
    elif lev_dim in T.coords:
        p = T[lev_dim]
    else:
        raise ValueError("Could not determine pressure. Provide 'pressure', or "
                         "(hyam, hybm, ps) for hybrid levels, or ensure the vertical "
                         "coordinate holds pressure values.")
    #----------------------------------------------------------------------------
    p = _to_pascals(p)
    return p.broadcast_like(T)
#---------------------------------------------------------------------------------------------------
def _build_mixing_ratio(qv, T, p, humidity_type):
    """ Return the water vapor mixing ratio [kg/kg] from the supplied humidity. """
    ht = humidity_type.lower()
    if ht in ('specific','q','sh','specific_humidity'):
        return qv / (1.0 - qv)
    if ht in ('mixing','mixing_ratio','r','mr'):
        return qv
    if ht in ('relative','rh','relative_humidity'):
        rh = qv
        if float(np.nanmax(rh.values if isinstance(rh, xr.DataArray) else rh)) > 1.5:
            rh = rh / 100.0   # convert percent to fraction
        es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        rs = eps * es / (p - es)
        return rh * rs
    raise ValueError(f"Unknown humidity_type '{humidity_type}'. Expected 'specific', "
                     f"'mixing', or 'relative'.")
#---------------------------------------------------------------------------------------------------
def calc_cape(T, qv, pressure=None, ps=None, hyam=None, hybm=None, p0=None,
              lev_dim=None, humidity_type='specific', virtual=True,
              return_cin=False, n_substeps=20):
    """ Compute undilute (pseudoadiabatic) surface-based CAPE.

    A surface parcel is lifted dry-adiabatically to its LCL and pseudoadiabatically
    above it; the virtual-temperature buoyancy is integrated in log-pressure. Any
    time and horizontal dimensions are preserved -- only the vertical dimension is
    collapsed.

    Parameters
    ----------
    T : xr.DataArray
        Air temperature [K] with a vertical dimension.
    qv : xr.DataArray
        Humidity, interpreted according to `humidity_type`.
    pressure : xr.DataArray or array-like, optional
        Pressure at each level [Pa or hPa]. Supply this for pressure-level data or
        for height-level data (where pressure cannot be derived from the vertical
        coordinate). A 1-D array is assumed to lie along the vertical dimension.
    ps, hyam, hybm, p0 : optional
        E3SM hybrid-level inputs. When `pressure` is not given, pressure is built as
        p = hyam*p0 + hybm*ps. `p0` defaults to 1e5 Pa. `ps` may be Pa or hPa.
    lev_dim : str, optional
        Name of the vertical dimension. Auto-detected from common names if omitted.
    humidity_type : {'specific','mixing','relative'}
        Interpretation of `qv`. Relative humidity may be a fraction or percent.
    virtual : bool
        Use virtual temperature for buoyancy (recommended). Default True.
    return_cin : bool
        If True, return an xr.Dataset with both 'cape' and 'cin'; otherwise return
        the CAPE DataArray only.
    n_substeps : int
        Substeps used to integrate the parcel ascent between adjacent levels.

    Returns
    -------
    xr.DataArray or xr.Dataset
        CAPE [J/kg] (and CIN [J/kg], a non-positive value, when return_cin=True),
        with the vertical dimension removed and all other dimensions preserved.
    """
    #----------------------------------------------------------------------------
    # resolve the vertical dimension and build aligned pressure / mixing-ratio fields
    #----------------------------------------------------------------------------
    lev_dim = _find_vert_dim(T, lev_dim)
    p = _build_pressure(T, lev_dim, pressure, ps, hyam, hybm, p0)
    r = _build_mixing_ratio(qv, T, p, humidity_type).broadcast_like(T)
    #----------------------------------------------------------------------------
    # move the vertical axis last and flatten every other dimension into columns
    #----------------------------------------------------------------------------
    lev_ax     = T.dims.index(lev_dim)
    other_dims = [d for d in T.dims if d != lev_dim]

    T_vals = np.moveaxis(np.asarray(T.values, dtype=np.float64),           lev_ax, -1)
    p_vals = np.moveaxis(np.asarray(p.transpose(*T.dims).values, np.float64), lev_ax, -1)
    r_vals = np.moveaxis(np.asarray(r.transpose(*T.dims).values, np.float64), lev_ax, -1)

    col_shape = T_vals.shape[:-1]
    nlev      = T_vals.shape[-1]
    ncols     = int(np.prod(col_shape)) if col_shape else 1

    T_2d = T_vals.reshape(ncols, nlev)
    p_2d = p_vals.reshape(ncols, nlev)
    r_2d = r_vals.reshape(ncols, nlev)
    #----------------------------------------------------------------------------
    # run the numba core over all columns
    #----------------------------------------------------------------------------
    cape_flat, cin_flat = _cape_cin_field(p_2d, T_2d, r_2d, bool(virtual), int(n_substeps))
    #----------------------------------------------------------------------------
    # rebuild output DataArrays, preserving the non-vertical dims and coords
    #----------------------------------------------------------------------------
    cape_arr = cape_flat.reshape(col_shape) if col_shape else cape_flat.reshape(())
    cin_arr  = cin_flat.reshape(col_shape)  if col_shape else cin_flat.reshape(())

    coords = {k: v for k, v in T.coords.items()
              if lev_dim not in v.dims and k != lev_dim}

    cape = xr.DataArray(cape_arr, dims=other_dims, coords=coords, name='cape',
                        attrs={'long_name':'convective available potential energy (undilute)',
                               'units':'J/kg'})
    if not return_cin:
        return cape
    cin = xr.DataArray(cin_arr, dims=other_dims, coords=coords, name='cin',
                       attrs={'long_name':'convective inhibition (undilute)',
                              'units':'J/kg'})
    return xr.Dataset({'cape': cape, 'cin': cin})
#---------------------------------------------------------------------------------------------------
