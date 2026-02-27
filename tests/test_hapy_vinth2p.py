"""
Unit tests for hapy_vinth2p.py

Run with:
    conda run -n ux_env python -m pytest tests/test_hapy_vinth2p.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from hapy_vinth2p import vinth2p, vinth2p_simple, _interp_columns_scipy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_pure_b_setup(nlev=5, ncol=1):
    """
    Return (hbcofa, hbcofb, psfc, p0) where pressure at each level is
    purely determined by hbcofb * psfc, giving well-known pressure values:
        p = [1.0, 0.8, 0.6, 0.4, 0.2] * psfc
    For psfc = 100000 Pa:
        p = [100000, 80000, 60000, 40000, 20000] Pa
    """
    p0    = 100000.0
    psfc  = np.full(ncol, 100000.0)
    hbcofa = np.zeros(nlev)
    hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:nlev]
    return hbcofa, hbcofb, psfc, p0


# ---------------------------------------------------------------------------
# _interp_columns_scipy
# ---------------------------------------------------------------------------

class TestInterpColumnsScipy:
    """Low-level column interpolation (scipy backend)."""

    def test_linear_exact(self):
        """Interpolate onto existing pressure levels → exact values."""
        n, nlevi, nlevo = 1, 5, 3
        # pressure decreasing surface→top: 100000, 80000, 60000, 40000, 20000
        p = np.array([[100000., 80000., 60000., 40000., 20000.]])
        # data linear in pressure
        data = p.copy() / 1000.           # 100, 80, 60, 40, 20
        plevo = np.array([100000., 60000., 20000.])
        datao = np.full((n, nlevo), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=1, kxtrp=False)
        npt.assert_allclose(result[0], [100., 60., 20.])

    def test_log_interpolation(self):
        """Log interpolation (intyp=2) on log-linear field."""
        # p = [e^5, e^4, e^3, e^2, e^1]; data = [5, 4, 3, 2, 1]
        p_vals = np.exp(np.array([5., 4., 3., 2., 1.]))
        data   = np.array([[5., 4., 3., 2., 1.]])
        p      = p_vals.reshape(1, -1)
        # Interpolate at e^3.5 → expect ~3.5
        plevo  = np.array([np.exp(3.5)])
        datao  = np.full((1, 1), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=2, kxtrp=False)
        npt.assert_allclose(result[0, 0], 3.5, rtol=1e-6)

    def test_out_of_range_nan(self):
        """Points outside column range → NaN when kxtrp=False."""
        p    = np.array([[90000., 70000., 50000.]])
        data = np.array([[9., 7., 5.]])
        plevo = np.array([95000., 45000.])  # both outside range
        datao = np.full((1, 2), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=1, kxtrp=False)
        assert np.all(np.isnan(result))

    def test_extrapolation(self):
        """kxtrp=True extrapolates linearly beyond the column."""
        p    = np.array([[100000., 80000.]])
        data = np.array([[10., 8.]])
        # slope = (8-10)/(80000-100000) = 1e-4 per Pa
        plevo = np.array([60000.])   # beyond top (80000)
        datao = np.full((1, 1), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=1, kxtrp=True)
        # expected: 8 + 1e-4*(60000-80000) = 8 + (-2) = 6
        npt.assert_allclose(result[0, 0], 6.0, rtol=1e-6)

    def test_nan_data_skipped(self):
        """Column with NaN data is left as NaN in output."""
        p    = np.array([[100000., 80000., 60000.]])
        data = np.array([[np.nan, 7., 5.]])
        plevo = np.array([80000.])
        datao = np.full((1, 1), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=1, kxtrp=False)
        assert np.isnan(result[0, 0])

    def test_multiple_columns(self):
        """Vectorised over multiple spatial columns."""
        p    = np.array([[100000., 50000.], [100000., 50000.]])
        data = np.array([[100., 50.], [200., 100.]])
        plevo = np.array([75000.])
        datao = np.full((2, 1), np.nan)
        result = _interp_columns_scipy(data, p, datao, plevo, intyp=1, kxtrp=False)
        npt.assert_allclose(result[:, 0], [75., 150.])

    def test_invalid_intyp_raises(self):
        """intyp not in {1,2,3} should raise ValueError."""
        p    = np.array([[100000., 80000.]])
        data = np.array([[10., 8.]])
        plevo = np.array([90000.])
        datao = np.full((1, 1), np.nan)
        with pytest.raises(ValueError, match="Invalid intyp"):
            _interp_columns_scipy(data, p, datao, plevo, intyp=99, kxtrp=False)


# ---------------------------------------------------------------------------
# vinth2p  (full function)
# ---------------------------------------------------------------------------

class TestVinth2p:

    def _simple_inputs(self, nlev=5):
        """Single-column (1, nlev) unstructured inputs with linear data."""
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:nlev]
        psfc   = np.array([100000.0])
        p0     = 100000.0
        # p levels: 100000, 80000, 60000, 40000, 20000 Pa
        data   = np.array([[100., 80., 60., 40., 20.]])  # = p/1000
        return data, hbcofa, hbcofb, psfc, p0

    def test_output_shape_unstructured(self):
        data, hbcofa, hbcofb, psfc, p0 = self._simple_inputs()
        plevo = np.array([90000., 70000., 50000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=None, kxtrp=False, use_numba=False)
        assert out.shape == (1, 3)

    def test_linear_interpolation_values(self):
        data, hbcofa, hbcofb, psfc, p0 = self._simple_inputs()
        plevo = np.array([90000., 70000., 50000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=None, kxtrp=False, use_numba=False)
        npt.assert_allclose(out[0], [90., 70., 50.], rtol=1e-5)

    def test_output_shape_structured(self):
        """Structured (lev, lat, lon) input."""
        nlev, nlat, nlon = 5, 4, 6
        hbcofa = np.zeros(nlev)
        hbcofb = np.linspace(1, 0.2, nlev)
        p0     = 100000.0
        psfc   = np.full((nlat, nlon), 100000.0)
        data   = np.random.randn(nlev, nlat, nlon)
        plevo  = np.array([90000., 70000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=0, kxtrp=False, use_numba=False)
        assert out.shape == (2, nlat, nlon)

    def test_4d_time_lev_lat_lon(self):
        ntime, nlev, nlat, nlon = 2, 5, 3, 4
        hbcofa = np.zeros(nlev)
        hbcofb = np.linspace(1.0, 0.2, nlev)
        p0     = 100000.0
        psfc   = np.full((ntime, nlat, nlon), 100000.0)
        data   = np.ones((ntime, nlev, nlat, nlon))
        plevo  = np.array([80000., 60000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=1, kxtrp=False, use_numba=False)
        assert out.shape == (ntime, 2, nlat, nlon)

    def test_mismatched_hyb_coef_raises(self):
        data, hbcofa, hbcofb, psfc, p0 = self._simple_inputs(nlev=5)
        bad_cofa = np.zeros(3)   # wrong length
        bad_cofb = np.zeros(3)
        with pytest.raises(ValueError, match="Hybrid coefficients"):
            vinth2p(data, bad_cofa, bad_cofb, np.array([90000.]), psfc,
                    intyp=1, p0=p0, ii=None, kxtrp=False, use_numba=False)

    def test_mismatched_psfc_raises(self):
        data, hbcofa, hbcofb, _, p0 = self._simple_inputs(nlev=5)
        bad_psfc = np.array([100000., 95000.])   # wrong spatial shape
        with pytest.raises(ValueError):
            vinth2p(data, hbcofa, hbcofb, np.array([90000.]), bad_psfc,
                    intyp=1, p0=p0, ii=None, kxtrp=False, use_numba=False)

    def test_no_extrapolation_out_of_range_nan(self):
        data, hbcofa, hbcofb, psfc, p0 = self._simple_inputs()
        # 5000 Pa is well above top level (20000 Pa)
        plevo = np.array([5000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=None, kxtrp=False, use_numba=False)
        assert np.isnan(out[0, 0])

    def test_xarray_input_returns_xarray(self):
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        psfc   = np.array([100000.0])
        p0     = 100000.0
        data_xr = xr.DataArray(
            np.array([[100., 80., 60., 40., 20.]]),
            dims=["ncol", "lev"],
            coords={"lev": np.arange(nlev, dtype=float)},
            attrs={"units": "K"},
            name="T",
        )
        plevo = np.array([90000., 70000.])
        out = vinth2p(data_xr, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=None, kxtrp=False, use_numba=False)
        assert isinstance(out, xr.DataArray)

    def test_xarray_output_dim_is_plev(self):
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        psfc   = np.array([100000.0])
        p0     = 100000.0
        data_xr = xr.DataArray(
            np.array([[100., 80., 60., 40., 20.]]),
            dims=["ncol", "lev"],
            coords={"lev": np.arange(nlev, dtype=float)},
        )
        plevo = np.array([90000., 70000.])
        out = vinth2p(data_xr, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=p0,
                      ii=None, kxtrp=False, use_numba=False)
        assert "plev" in out.dims
        assert "lev" not in out.dims

    def test_xarray_preserves_attrs_and_name(self):
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        psfc   = np.array([100000.0])
        data_xr = xr.DataArray(
            np.array([[100., 80., 60., 40., 20.]]),
            dims=["ncol", "lev"],
            coords={"lev": np.arange(nlev, dtype=float)},
            attrs={"units": "K"},
            name="temperature",
        )
        out = vinth2p(data_xr, hbcofa, hbcofb, np.array([90000.]), psfc,
                      intyp=1, p0=100000., ii=None, kxtrp=False, use_numba=False)
        assert out.attrs["units"] == "K"
        assert out.name == "temperature"

    def test_xarray_hyb_coef_with_time_dim(self):
        """Hybrid coefficients with a leading time dim are handled gracefully."""
        nlev = 5
        hbcofb_raw = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        hbcofa_xr = xr.DataArray(
            np.broadcast_to(np.zeros(nlev), (3, nlev)).copy(),
            dims=["time", "lev"],
        )
        hbcofb_xr = xr.DataArray(
            np.broadcast_to(hbcofb_raw, (3, nlev)).copy(),
            dims=["time", "lev"],
        )
        psfc = np.array([100000.0])
        data = np.array([[100., 80., 60., 40., 20.]])
        plevo = np.array([90000.])
        out = vinth2p(data, hbcofa_xr, hbcofb_xr, plevo, psfc, intyp=1,
                      p0=100000., ii=None, kxtrp=False, use_numba=False)
        npt.assert_allclose(out[0, 0], 90., rtol=1e-5)

    def test_constant_field_returns_constant(self):
        """If data is constant over levels, output should be that constant."""
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.linspace(1.0, 0.2, nlev)
        psfc   = np.array([100000.0])
        data   = np.full((1, nlev), 42.0)
        plevo  = np.array([90000., 60000., 30000.])
        out = vinth2p(data, hbcofa, hbcofb, plevo, psfc, intyp=1, p0=100000.,
                      ii=None, kxtrp=False, use_numba=False)
        npt.assert_allclose(out[0], 42.0)


# ---------------------------------------------------------------------------
# vinth2p_simple
# ---------------------------------------------------------------------------

class TestVinth2pSimple:

    def _inputs(self):
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        psfc   = np.array([100000.0])
        data   = np.array([[100., 80., 60., 40., 20.]])
        return data, hbcofa, hbcofb, psfc

    def test_linear_interface(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, [90000., 70000.], psfc,
                             interp_type="linear", use_numba=False)
        npt.assert_allclose(out[0], [90., 70.], rtol=1e-5)

    def test_log_interface(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, [90000.], psfc,
                             interp_type="log", use_numba=False)
        assert not np.isnan(out[0, 0])

    def test_loglog_interface(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, [90000.], psfc,
                             interp_type="loglog", use_numba=False)
        assert not np.isnan(out[0, 0])

    def test_unknown_interp_type_defaults_to_log(self):
        """An unrecognised interp_type string should not raise – defaults to log."""
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, [90000.], psfc,
                             interp_type="unknown", use_numba=False)
        assert out.shape == (1, 1)

    def test_scalar_plevo(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, 90000., psfc,
                             interp_type="linear", use_numba=False)
        assert out.shape == (1, 1)

    def test_extrapolate_flag(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        # Request a level above the model top (20000 Pa)
        plevo = [5000.]
        out_no_extrap = vinth2p_simple(data, hbcofa, hbcofb, plevo, psfc,
                                       extrapolate=False, use_numba=False)
        out_extrap    = vinth2p_simple(data, hbcofa, hbcofb, plevo, psfc,
                                       extrapolate=True,  use_numba=False)
        assert np.isnan(out_no_extrap[0, 0])
        assert not np.isnan(out_extrap[0, 0])

    def test_returns_same_type_as_input_numpy(self):
        data, hbcofa, hbcofb, psfc = self._inputs()
        out = vinth2p_simple(data, hbcofa, hbcofb, [90000.], psfc,
                             use_numba=False)
        assert isinstance(out, np.ndarray)

    def test_returns_xarray_for_xarray_input(self):
        nlev = 5
        hbcofa = np.zeros(nlev)
        hbcofb = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        psfc   = np.array([100000.0])
        data_xr = xr.DataArray(
            np.array([[100., 80., 60., 40., 20.]]),
            dims=["ncol", "lev"],
            coords={"lev": np.arange(nlev, dtype=float)},
        )
        out = vinth2p_simple(data_xr, hbcofa, hbcofb, [90000., 70000.], psfc,
                             use_numba=False)
        assert isinstance(out, xr.DataArray)
