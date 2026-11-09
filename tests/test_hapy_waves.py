"""
Unit tests for hapy_waves.py – native grid Wheeler-Kiladis spectra.

Run with:
    conda run -n ux_env pytest tests/test_hapy_waves.py -v
"""
import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from hapy_waves import zonal_fourier_native, spacetime_power_from_zonal_fourier, spacetime_power_native

wf = pytest.importorskip("wavenumber_frequency_functions")
logging.disable(logging.CRITICAL)  # wf sets the logging level to DEBUG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latlon(ntime=200, nlat=13, nlon=144, seed=0):
    """Random field plus an eastward propagating wave on a regular lat/lon grid."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-15, 15, nlat)
    lon = np.arange(nlon) * 360. / nlon
    t = np.arange(ntime)
    wave = np.cos(np.deg2rad(5 * lon)[None, None, :] - 2 * np.pi * t[:, None, None] / 10.)
    wave = wave * np.exp(-(lat[None, :, None] / 10.)**2)
    vals = 240. + 10. * wave + rng.standard_normal((ntime, nlat, nlon))
    return xr.DataArray(vals, dims=("time", "lat", "lon"),
                        coords={"time": t, "lat": lat, "lon": lon})


def _as_columns(da):
    """Flatten a (time,lat,lon) DataArray into (time,ncol) plus column lat/lon/area."""
    LON, LAT = np.meshgrid(da["lon"].values, da["lat"].values)
    flat = da.stack(ncol=("lat", "lon")).drop_vars(["ncol", "lat", "lon"]).transpose("time", "ncol")
    return flat, LAT.ravel(), LON.ravel(), np.cos(np.deg2rad(LAT)).ravel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zonal_fourier_matches_fft():
    """On a regular grid the projection should equal np.fft.fft(...)/nlon."""
    da = _make_latlon(ntime=5)
    flat, lat, lon, area = _as_columns(da)
    kmax = 20
    X = zonal_fourier_native(flat, lat, lon, area, lat2=15, dlat=2.5, kmax=kmax, verbose=False)
    ref = np.fft.fft(da.values, axis=-1) / da.sizes["lon"]
    wvn = X["wavenumber"].values.astype(int)
    npt.assert_allclose(X.values, ref[:, :, wvn], rtol=0, atol=1e-10)


@pytest.mark.parametrize("kmax", [72, 30])
def test_spectra_match_spacetime_power(kmax):
    """lat/lon data treated as unstructured columns should reproduce wf.spacetime_power."""
    da = _make_latlon()
    segsize, noverlap = 96, 60
    ref = wf.spacetime_power(da, segsize=segsize, noverlap=noverlap, spd=1, dosymmetries=True)
    flat, lat, lon, area = _as_columns(da)
    spec = spacetime_power_native(flat, lat, lon, area, lat2=15, dlat=2.5, kmax=kmax,
                                  segsize=segsize, noverlap=noverlap, spd=1, verbose=False)
    k = slice(-kmax + 1, kmax - 1)
    a = ref.sel(wavenumber=k).values
    b = spec.sel(wavenumber=k).values
    npt.assert_allclose(b, a, rtol=1e-8, atol=1e-10 * np.nanmax(a))


def test_wave_peak_location():
    """The imposed k=5, 10-day eastward wave should be the symmetric spectral peak."""
    da = _make_latlon()
    flat, lat, lon, area = _as_columns(da)
    spec = spacetime_power_native(flat, lat, lon, area, lat2=15, dlat=2.5, kmax=30,
                                  segsize=96, noverlap=60, spd=1, verbose=False)
    sym = spec.sel(component="symmetric").sel(frequency=slice(0.02, None))
    imax = np.unravel_index(np.nanargmax(sym.values), sym.shape)
    assert sym["wavenumber"].values[imax[0]] == 5
    npt.assert_allclose(sym["frequency"].values[imax[1]], 0.1, atol=1. / 96)


def test_empty_bin_raises():
    da = _make_latlon(ntime=5)
    flat, lat, lon, area = _as_columns(da)
    with pytest.raises(ValueError, match="empty latitude bin"):
        zonal_fourier_native(flat, lat, lon, area, lat2=15, dlat=1.0, kmax=10, verbose=False)


def test_short_series_raises():
    da = _make_latlon(ntime=50)
    flat, lat, lon, area = _as_columns(da)
    X = zonal_fourier_native(flat, lat, lon, area, lat2=15, dlat=2.5, kmax=10, verbose=False)
    with pytest.raises(ValueError, match="not enough time samples"):
        spacetime_power_from_zonal_fourier(X, segsize=96, noverlap=60, spd=1)
