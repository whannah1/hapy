"""
Unit tests for hapy_interp.py – focused on interp_to_height().

Run with:
    conda run -n ux_env pytest tests/test_hapy_interp.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from hapy_interp import interp_to_height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_da(values, lev, lev_dim="lev", name="T"):
    """Return a 1-D DataArray along lev_dim."""
    return xr.DataArray(values, dims=[lev_dim], coords={lev_dim: lev}, name=name)


def _make_zgeo(values, lev, lev_dim="lev"):
    return xr.DataArray(values, dims=[lev_dim], coords={lev_dim: lev})


# ---------------------------------------------------------------------------
# Basic interpolation (single column)
# ---------------------------------------------------------------------------

class TestSingleColumn:
    """1-D (lev,) inputs – simplest case."""

    lev   = np.array([1, 2, 3, 4, 5], dtype=float)
    z     = np.array([100., 200., 300., 400., 500.])   # height monotonically increasing
    vals  = np.array([10.,  20.,  30.,  40.,  50.])    # linear in z

    def _da(self):
        return _make_da(self.vals, self.lev)

    def _zgeo(self):
        return _make_zgeo(self.z, self.lev)

    def test_exact_levels(self):
        """Interpolating at existing height levels returns exact values."""
        result = interp_to_height(self._da(), self._zgeo(), self.z)
        npt.assert_allclose(result.values, self.vals)

    def test_midpoint_interpolation(self):
        """Midpoints between levels are linearly interpolated."""
        target = np.array([150., 250., 350., 450.])
        result = interp_to_height(self._da(), self._zgeo(), target)
        npt.assert_allclose(result.values, [15., 25., 35., 45.])

    def test_output_dim_name(self):
        """Output DataArray uses the requested height_dim name."""
        result = interp_to_height(self._da(), self._zgeo(), [200., 300.],
                                  height_dim="altitude")
        assert "altitude" in result.dims
        assert "lev" not in result.dims

    def test_output_coord_values(self):
        """Output coordinate stores the target heights."""
        target = [150., 350.]
        result = interp_to_height(self._da(), self._zgeo(), target)
        npt.assert_array_equal(result.coords["height"].values, target)

    def test_out_of_range_nan_by_default(self):
        """Values outside the column height range are NaN when extrapolate=False."""
        result = interp_to_height(self._da(), self._zgeo(),
                                  [50., 200., 600.])   # 50 below, 600 above
        assert np.isnan(result.values[0]), "below-range should be NaN"
        assert np.isnan(result.values[2]), "above-range should be NaN"
        npt.assert_allclose(result.values[1], 20.)

    def test_preserves_attrs_and_name(self):
        da = self._da()
        da.attrs = {"units": "K"}
        result = interp_to_height(da, self._zgeo(), [200., 300.])
        assert result.name == "T"
        assert result.attrs["units"] == "K"

    def test_lev_dim_removed_from_output(self):
        result = interp_to_height(self._da(), self._zgeo(), [200., 300.])
        assert "lev" not in result.dims

    def test_single_target_height(self):
        result = interp_to_height(self._da(), self._zgeo(), [300.])
        assert result.shape == (1,)
        npt.assert_allclose(result.values[0], 30.)


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------

class TestExtrapolation:
    lev  = np.array([1, 2, 3], dtype=float)
    z    = np.array([100., 200., 300.])
    vals = np.array([10.,  20.,  30.])     # slope = 0.1 val/m

    def _da(self):    return _make_da(self.vals, self.lev)
    def _zgeo(self):  return _make_zgeo(self.z, self.lev)

    def test_extrapolate_below(self):
        """Linear extrapolation below the column bottom."""
        result = interp_to_height(self._da(), self._zgeo(), [0.],
                                  extrapolate=True)
        # slope = (20-10)/(200-100) = 0.1 → at z=0: 10 + 0.1*(0-100) = 0
        npt.assert_allclose(result.values[0], 0.)

    def test_extrapolate_above(self):
        """Linear extrapolation above the column top."""
        result = interp_to_height(self._da(), self._zgeo(), [400.],
                                  extrapolate=True)
        # slope = (30-20)/(300-200) = 0.1 → at z=400: 30 + 0.1*(400-300) = 40
        npt.assert_allclose(result.values[0], 40.)

    def test_no_extrapolate_returns_nan(self):
        result = interp_to_height(self._da(), self._zgeo(), [0., 400.],
                                  extrapolate=False)
        assert np.all(np.isnan(result.values))


# ---------------------------------------------------------------------------
# Descending height order (top-of-atmosphere → surface, e.g. pressure-like)
# ---------------------------------------------------------------------------

class TestDescendingHeights:
    lev  = np.array([1, 2, 3, 4, 5], dtype=float)
    z    = np.array([500., 400., 300., 200., 100.])  # decreasing
    vals = np.array([50.,  40.,  30.,  20.,  10.])

    def _da(self):    return _make_da(self.vals, self.lev)
    def _zgeo(self):  return _make_zgeo(self.z, self.lev)

    def test_interpolation_descending(self):
        result = interp_to_height(self._da(), self._zgeo(), [150., 350.])
        npt.assert_allclose(result.values, [15., 35.])

    def test_out_of_range_nan_descending(self):
        result = interp_to_height(self._da(), self._zgeo(), [50., 300., 600.])
        assert np.isnan(result.values[0])
        assert np.isnan(result.values[2])
        npt.assert_allclose(result.values[1], 30.)


# ---------------------------------------------------------------------------
# Multi-column (ncol, lev)
# ---------------------------------------------------------------------------

class TestMultiColumn:
    lev  = np.array([1, 2, 3], dtype=float)
    z    = np.array([[100., 200., 300.],   # col 0
                     [100., 200., 300.]])  # col 1 – same heights
    vals = np.array([[10.,  20.,  30.],
                     [100., 200., 300.]])

    def _da(self):
        return xr.DataArray(self.vals, dims=["ncol", "lev"],
                            coords={"lev": self.lev})

    def _zgeo(self):
        return xr.DataArray(self.z, dims=["ncol", "lev"],
                            coords={"lev": self.lev})

    def test_shape(self):
        result = interp_to_height(self._da(), self._zgeo(), [150., 250.])
        assert result.shape == (2, 2)

    def test_values(self):
        result = interp_to_height(self._da(), self._zgeo(), [150., 250.])
        npt.assert_allclose(result.values[0], [15., 25.])
        npt.assert_allclose(result.values[1], [150., 250.])

    def test_dims(self):
        result = interp_to_height(self._da(), self._zgeo(), [150.])
        assert result.dims == ("ncol", "height")


# ---------------------------------------------------------------------------
# Time × ncol × lev
# ---------------------------------------------------------------------------

class TestTimeDimension:
    lev    = np.array([1, 2, 3], dtype=float)
    z_ncol = np.array([[100., 200., 300.]])   # (1 ncol, 3 lev) – broadcast over time
    # Two time steps, 1 column
    vals   = np.array([[[10., 20., 30.]],     # t=0
                       [[40., 50., 60.]]])    # t=1   (shape: time, ncol, lev)

    def _da(self):
        return xr.DataArray(self.vals, dims=["time", "ncol", "lev"],
                            coords={"lev": self.lev,
                                    "time": [0, 1]})

    def _zgeo(self):
        # zgeo without time dim – should be broadcast to (time, ncol, lev)
        return xr.DataArray(self.z_ncol, dims=["ncol", "lev"],
                            coords={"lev": self.lev})

    def test_shape(self):
        result = interp_to_height(self._da(), self._zgeo(), [150., 250.])
        assert result.shape == (2, 1, 2)   # (time, ncol, height)

    def test_values(self):
        result = interp_to_height(self._da(), self._zgeo(), [150., 250.])
        npt.assert_allclose(result.values[0, 0], [15., 25.])
        npt.assert_allclose(result.values[1, 0], [45., 55.])

    def test_dims(self):
        result = interp_to_height(self._da(), self._zgeo(), [150.])
        assert result.dims == ("time", "ncol", "height")


# ---------------------------------------------------------------------------
# Custom dim names
# ---------------------------------------------------------------------------

class TestCustomDimNames:
    def test_custom_lev_dim(self):
        lev  = np.array([1., 2., 3.])
        da   = xr.DataArray([10., 20., 30.], dims=["plev"], coords={"plev": lev})
        zgeo = xr.DataArray([100., 200., 300.], dims=["plev"], coords={"plev": lev})
        result = interp_to_height(da, zgeo, [150., 250.], lev_dim="plev")
        npt.assert_allclose(result.values, [15., 25.])

    def test_custom_height_dim(self):
        lev  = np.array([1., 2., 3.])
        da   = xr.DataArray([10., 20., 30.], dims=["lev"], coords={"lev": lev})
        zgeo = xr.DataArray([100., 200., 300.], dims=["lev"], coords={"lev": lev})
        result = interp_to_height(da, zgeo, [200.], height_dim="z")
        assert "z" in result.dims


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_field(self):
        """A field that is constant in height should return that constant everywhere."""
        lev  = np.arange(1., 6.)
        z    = np.array([100., 200., 300., 400., 500.])
        vals = np.full(5, 42.)
        da   = _make_da(vals, lev)
        zgeo = _make_zgeo(z, lev)
        result = interp_to_height(da, zgeo, [150., 250., 350.])
        npt.assert_allclose(result.values, 42.)

    def test_two_level_column(self):
        """Minimum valid case: two levels."""
        lev  = np.array([1., 2.])
        z    = np.array([0., 1000.])
        vals = np.array([0., 1000.])
        da   = _make_da(vals, lev)
        zgeo = _make_zgeo(z, lev)
        result = interp_to_height(da, zgeo, [500.])
        npt.assert_allclose(result.values[0], 500.)

    def test_target_heights_dtype_float64(self):
        """Output coordinate should always be float64."""
        lev  = np.arange(1., 4.)
        z    = np.array([100., 200., 300.])
        vals = np.array([1., 2., 3.])
        da   = _make_da(vals, lev)
        zgeo = _make_zgeo(z, lev)
        result = interp_to_height(da, zgeo, [150, 250])  # integer targets
        assert result.coords["height"].dtype == np.float64
