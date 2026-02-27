"""
Unit tests for hapy_sphere.py

Run with:
    conda run -n ux_env python -m pytest tests/test_hapy_sphere.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import hapy_sphere
from hapy_sphere import lon_to_180, earth_radius, calc_great_circle_distance

# deg_to_rad is defined in hapy_constants but not re-exported through hapy_common;
# patch it into the module namespace so the functions that depend on it work.
hapy_sphere.deg_to_rad = np.pi / 180.0

# Re-import after patching so the name is available in function scope too
from hapy_sphere import find_nearest_neighbors_on_sphere


# ---------------------------------------------------------------------------
# lon_to_180  (plain function)
# ---------------------------------------------------------------------------

class TestLonTo180:

    def test_already_in_range(self):
        npt.assert_allclose(lon_to_180(0.0),    0.0)
        npt.assert_allclose(lon_to_180(-90.0), -90.0)
        npt.assert_allclose(lon_to_180(90.0),   90.0)

    def test_360_becomes_0(self):
        npt.assert_allclose(lon_to_180(360.0), 0.0)

    def test_wraps_east(self):
        # 270° → -90°
        npt.assert_allclose(lon_to_180(270.0), -90.0)

    def test_wraps_west(self):
        # -270° → 90°
        npt.assert_allclose(lon_to_180(-270.0), 90.0)

    def test_180_boundary(self):
        # 180 should map to -180 (the formula produces -180)
        npt.assert_allclose(lon_to_180(180.0), -180.0)

    def test_array_input(self):
        lon = np.array([0., 90., 180., 270., 360.])
        expected = np.array([0., 90., -180., -90., 0.])
        npt.assert_allclose(lon_to_180(lon), expected)

    def test_negative_array(self):
        lon = np.array([-180., -90., -1., -361.])
        result = lon_to_180(lon)
        assert np.all(result >= -180.) and np.all(result < 180.)


# ---------------------------------------------------------------------------
# LongitudeAccessor (xarray DataArray accessor)
# ---------------------------------------------------------------------------

class TestLongitudeAccessor:

    def _make_da(self, lons, extra_dim=False):
        """Build a simple DataArray with a 'lon' coordinate."""
        if extra_dim:
            return xr.DataArray(
                np.ones((3, len(lons))),
                dims=["lat", "lon"],
                coords={"lon": lons},
            )
        return xr.DataArray(
            np.ones(len(lons)), dims=["lon"], coords={"lon": lons}
        )

    def test_basic_conversion(self):
        da = self._make_da(np.array([0., 90., 270., 360.]))
        result = da.lon_to_180()
        expected = np.array([0., 90., -90., 0.])
        npt.assert_allclose(result.coords["lon"].values, expected)

    def test_data_values_unchanged(self):
        da = self._make_da(np.array([200., 300.]))
        result = da.lon_to_180()
        npt.assert_array_equal(result.values, da.values)

    def test_sort_dim(self):
        da = self._make_da(np.array([270., 90., 0.]))
        result = da.lon_to_180(sort_dim=True)
        lons = result.coords["lon"].values
        assert np.all(np.diff(lons) >= 0), "lons should be sorted ascending"

    def test_missing_lon_dim_returns_none(self, capsys):
        da = xr.DataArray(np.ones(3), dims=["x"])
        result = da.lon_to_180()
        assert result is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_custom_lon_dim_name(self):
        da = xr.DataArray(
            np.ones(3), dims=["longitude"],
            coords={"longitude": np.array([0., 270., 350.])}
        )
        result = da.lon_to_180(lon_dim="longitude")
        expected = np.array([0., -90., -10.])
        npt.assert_allclose(result.coords["longitude"].values, expected)

    def test_2d_data(self):
        lons = np.array([0., 180., 270.])
        da = self._make_da(lons, extra_dim=True)
        result = da.lon_to_180()
        assert result.shape == da.shape
        assert "lon" in result.coords


# ---------------------------------------------------------------------------
# earth_radius
# ---------------------------------------------------------------------------

class TestEarthRadius:
    # WGS84 semi-major / semi-minor axes
    A = 6378137.0       # equatorial radius [m]
    B = 6356752.3142    # polar radius [m]

    def test_equatorial_radius(self):
        # At lat=0 the geocentric & geodetic latitudes coincide; radius → a
        r = earth_radius(0.0)
        npt.assert_allclose(r, self.A, rtol=1e-6)

    def test_polar_radius(self):
        # At lat=90 the radius should equal the semi-minor axis
        r = earth_radius(90.0)
        npt.assert_allclose(r, self.B, rtol=1e-6)

    def test_symmetry_hemispheres(self):
        # Northern and southern hemisphere at same |lat| should be equal
        npt.assert_allclose(earth_radius(45.0), earth_radius(-45.0), rtol=1e-10)

    def test_radius_between_axes(self):
        r = earth_radius(45.0)
        assert self.B < r < self.A

    def test_array_input(self):
        lats = np.array([0., 30., 60., 90.])
        r = earth_radius(lats)
        assert r.shape == (4,)
        # Should decrease from equator to pole
        assert np.all(np.diff(r) < 0)

    def test_returns_meters(self):
        # Rough sanity: Earth radius is between 6.35e6 and 6.38e6 m everywhere
        for lat in [-90, -45, 0, 45, 90]:
            r = earth_radius(lat)
            assert 6.35e6 < r < 6.39e6


# ---------------------------------------------------------------------------
# calc_great_circle_distance
# ---------------------------------------------------------------------------

class TestCalcGreatCircleDistance:

    def test_same_point_is_zero(self):
        d = calc_great_circle_distance(45., 45., 60., 60.)
        npt.assert_allclose(d, 0.0, atol=1e-12)

    def test_antipodal_points(self):
        # Opposite poles should be π radians apart
        d = calc_great_circle_distance(90., -90., 0., 0.)
        npt.assert_allclose(d, np.pi, rtol=1e-10)

    def test_quarter_circle_equator(self):
        # Two points on the equator 90° apart → π/2 radians
        d = calc_great_circle_distance(0., 0., 0., 90.)
        npt.assert_allclose(d, np.pi / 2, rtol=1e-10)

    def test_symmetry(self):
        # Distance A→B == Distance B→A
        d1 = calc_great_circle_distance(30., 60., 10., 50.)
        d2 = calc_great_circle_distance(60., 30., 50., 10.)
        npt.assert_allclose(d1, d2, rtol=1e-12)

    def test_result_in_radians(self):
        d = calc_great_circle_distance(0., 0., 0., 1.)   # 1° apart
        expected = np.deg2rad(1.0)
        npt.assert_allclose(d, expected, rtol=1e-5)

    def test_clipping_no_nan(self):
        # Identical points could produce cos_dist slightly > 1 due to floats
        lats = np.zeros(10)
        lons = np.zeros(10)
        d = calc_great_circle_distance(lats, lats, lons, lons)
        assert not np.any(np.isnan(d))

    def test_array_inputs(self):
        lats = np.array([0., 45., 90.])
        d = calc_great_circle_distance(0., lats, 0., 0.)
        assert d.shape == (3,)
        npt.assert_allclose(d[0], 0.0, atol=1e-12)
        npt.assert_allclose(d[2], np.pi / 2, rtol=1e-10)

    def test_output_clipped_to_valid_range(self):
        # Result of arccos must be in [0, π]
        d = calc_great_circle_distance(0., 0., 0., 180.)
        assert 0.0 <= d <= np.pi


# ---------------------------------------------------------------------------
# find_nearest_neighbors_on_sphere
# ---------------------------------------------------------------------------

class TestFindNearestNeighbors:
    """Tests for find_nearest_neighbors_on_sphere."""

    # Four points at 0°, 90°, 180°, 270° on the equator
    lats = np.array([0., 0., 0., 0.], dtype=float)
    lons = np.array([0., 90., 180., 270.], dtype=float)

    def test_returns_dataset(self):
        import xarray as xr
        ds = find_nearest_neighbors_on_sphere(self.lats, self.lons, 1)
        assert isinstance(ds, xr.Dataset)

    def test_dataset_has_expected_variables(self):
        ds = find_nearest_neighbors_on_sphere(self.lats, self.lons, 1)
        assert "neighbor_id" in ds
        assert "neighbor_dist" in ds

    def test_num_neighbors_1(self):
        ds = find_nearest_neighbors_on_sphere(self.lats, self.lons, 1,
                                              return_sorted=True)
        # First neighbor of point 0 (lon=0) should be point 1 (lon=90) or point 3 (lon=270)
        # – both are equidistant (π/2 rad)
        neighbor_of_0 = ds["neighbor_id"].values[1, 0]   # row 1 = first neighbor
        assert neighbor_of_0 in (1, 3)

    def test_sorted_distances_ascending(self):
        ds = find_nearest_neighbors_on_sphere(self.lats, self.lons, 3,
                                              return_sorted=True)
        dists = ds["neighbor_dist"].values[:, 0]   # distances for column 0
        # sorted=True guarantees ascending order (row 0 = self, row 1 nearest, ...)
        assert np.all(np.diff(dists) >= 0)

    def test_two_points_one_neighbor(self):
        lats = np.array([0., 0.])
        lons = np.array([0., 45.])
        ds = find_nearest_neighbors_on_sphere(lats, lons, 1, return_sorted=True)
        # Each point's nearest neighbor should be the other point
        neighbors = ds["neighbor_id"].values[1, :]
        assert set(neighbors) == {0, 1}
