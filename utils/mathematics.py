import numpy as np
from scipy.interpolate import griddata
import math

# -----------------------------------------------------------------------
# Module: mathematics.py
# Purpose: Mathematical utilities for oceanographic data processing,
#          including spatial interpolation, statistical error metrics,
#          and differential operators (vorticity, divergence, kinetic energy).
# -----------------------------------------------------------------------


def interp_grid_vel(lon_high_res_grid, lat_high_res_grid, u_high_res_grid, v_high_res_grid,
                    lon_coarse_grid, lat_coarse_grid, mask_coarse_grid=[], points_mode=False):
    """
    Interpolates velocity components (u, v) from a high-resolution grid to a
    coarser grid (or to a set of scattered points) using linear interpolation.

    Parameters:
    -----------
    lon_high_res_grid, lat_high_res_grid : ndarray
        Longitude and latitude of the high-resolution source grid
    u_high_res_grid, v_high_res_grid : ndarray
        East and north velocity components on the high-resolution grid
    lon_coarse_grid, lat_coarse_grid : ndarray
        Target grid coordinates (1D or 2D); or 1D point arrays if points_mode=True
    mask_coarse_grid : ndarray, optional
        Land/sea mask applied to the interpolated values (only in grid mode)
    points_mode : bool
        If True, interpolate to irregular scattered points (1D arrays);
        if False, interpolate to a regular or 2D grid

    Returns:
    --------
    u_interp, v_interp : ndarray
        Interpolated east and north velocity components on the target grid/points
    """
    # Flatten source grid to a list of (lon, lat) points
    points = np.column_stack((lon_high_res_grid.flatten(), lat_high_res_grid.flatten()))
    u_values = u_high_res_grid.flatten()
    v_values = v_high_res_grid.flatten()

    # Remove NaN values from the source data before interpolating
    valid_indices = ~np.isnan(u_values) & ~np.isnan(v_values)
    valid_points = points[valid_indices]
    valid_u = u_values[valid_indices]
    valid_v = v_values[valid_indices]

    # Handle masked arrays by filling masked values with NaN
    if hasattr(valid_u, 'filled'):
        valid_u = valid_u.filled(np.nan)
    if hasattr(valid_v, 'filled'):
        valid_v = valid_v.filled(np.nan)

    if points_mode:
        # Interpolate to irregular 1D scattered points
        xi = np.column_stack((lon_coarse_grid, lat_coarse_grid))
        shape_out = lon_coarse_grid.shape

    else:
        # Build 2D meshgrid if 1D coordinate arrays are provided
        if lon_coarse_grid.ndim == 1 and lat_coarse_grid.ndim == 1:
            lon_mesh_coarse_grid, lat_mesh_coarse_grid = np.meshgrid(lon_coarse_grid, lat_coarse_grid)
        else:
            lon_mesh_coarse_grid, lat_mesh_coarse_grid = lon_coarse_grid, lat_coarse_grid

        xi = np.column_stack((lon_mesh_coarse_grid.flatten(), lat_mesh_coarse_grid.flatten()))
        shape_out = lon_mesh_coarse_grid.shape

    # Perform linear interpolation; points outside the convex hull become NaN
    u_interp = griddata(valid_points, valid_u, xi, method='linear', fill_value=np.nan)
    v_interp = griddata(valid_points, valid_v, xi, method='linear', fill_value=np.nan)

    # Reshape to the target grid shape
    u_interp = u_interp.reshape(shape_out)
    v_interp = v_interp.reshape(shape_out)

    # Apply the land/sea mask (sets land points to zero)
    if len(mask_coarse_grid) != 0 and not points_mode:
        u_interp = u_interp * mask_coarse_grid
        v_interp = v_interp * mask_coarse_grid

    return u_interp, v_interp


# --------------------------------------------------------------


def interp_grid_eta(lon_high_res_grid, lat_high_res_grid, eta_high_res_grid,
                    lon_coarse_grid, lat_coarse_grid, mask_coarse_grid=[]):
    """
    Interpolates the sea surface height (SSH / eta) from a high-resolution grid
    to a coarser grid using linear interpolation.

    Parameters:
    -----------
    lon_high_res_grid, lat_high_res_grid : ndarray
        Longitude and latitude of the high-resolution source grid
    eta_high_res_grid : ndarray
        Sea surface height on the high-resolution grid
    lon_coarse_grid, lat_coarse_grid : ndarray
        Target grid coordinates (1D or 2D)
    mask_coarse_grid : ndarray, optional
        Land/sea mask applied to the interpolated SSH values

    Returns:
    --------
    eta_interp : ndarray
        Interpolated SSH on the target grid
    """
    # Build 2D meshgrid if 1D coordinate arrays are provided
    if lon_coarse_grid.ndim == 1 and lat_coarse_grid.ndim == 1:
        lon_mesh_coarse_grid, lat_mesh_coarse_grid = np.meshgrid(lon_coarse_grid, lat_coarse_grid)
    else:
        lon_mesh_coarse_grid, lat_mesh_coarse_grid = lon_coarse_grid, lat_coarse_grid

    # Flatten the source grid to (lon, lat) point pairs
    points = np.column_stack((lon_high_res_grid.flatten(), lat_high_res_grid.flatten()))

    eta_values = eta_high_res_grid.flatten()

    # Remove NaN values from the source data
    valid_indices = ~np.isnan(eta_values)
    valid_points = points[valid_indices]
    valid_eta = eta_values[valid_indices]

    xi = np.column_stack((lon_mesh_coarse_grid.flatten(), lat_mesh_coarse_grid.flatten()))

    eta_interp = griddata(valid_points, valid_eta, xi, method='linear', fill_value=np.nan)
    eta_interp = eta_interp.reshape(lon_mesh_coarse_grid.shape)

    # Apply the land/sea mask
    if len(mask_coarse_grid) != 0:
        eta_interp = eta_interp * mask_coarse_grid

    return eta_interp


# --------------------------------------------------------------

def putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                               u_array, v_array):
    """
    Assigns scattered data points to the nearest node of a Least Squares (LS) grid.
    Points that do not fall exactly on a grid node are flagged with a warning.

    Parameters:
    -----------
    lon_array : array-like
        Longitudes of the data points
    lat_array : array-like
        Latitudes of the data points
    lon_grid_LS : ndarray
        2D longitude grid of the LS mesh
    lat_grid_LS : ndarray
        2D latitude grid of the LS mesh
    u_array : array-like
        East velocity component at each data point
    v_array : array-like
        North velocity component at each data point

    Returns:
    --------
    u_LS_2d, v_LS_2d : ndarray
        East and north velocity components mapped onto the 2D LS grid (NaN elsewhere)
    """
    tolerance = 1e-6

    # Initialise output arrays with NaN
    u_LS_2d = np.full(lon_grid_LS.shape, np.nan)
    v_LS_2d = np.full(lon_grid_LS.shape, np.nan)

    points_found = 0
    points_not_found = 0

    for j in range(len(lon_array)):
        # Find the closest grid node using Euclidean distance in degree space
        distances = np.sqrt((lon_grid_LS - lon_array[j])**2 +
                            (lat_grid_LS - lat_array[j])**2)
        min_distance = np.min(distances)
        idx_closest = np.unravel_index(np.argmin(distances), distances.shape)
        idx_lon, idx_lat = idx_closest

        # Verify that the closest node matches the data point within tolerance
        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance and
                         abs(lat_closest - lat_array[j]) < tolerance)

        if is_grid_point:
            points_found += 1
        else:
            points_not_found += 1

            if points_not_found <= 5:  # Show only the first 5 warnings
                print(f"WARNING: Point {j} is not exactly on the grid")
                print(f"  Distance: {min_distance:.6f}")
                print(f"  Target:   ({lon_array[j]:.6f}, {lat_array[j]:.6f})")
                print(f"  Closest:  ({lon_closest:.6f}, {lat_closest:.6f})")

        # Assign velocities to the closest grid node
        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]

    print(f"\nSummary: {points_found} points found, {points_not_found} points not exact")

    return u_LS_2d, v_LS_2d


# --------------------------------------------------------------


def stat_err(u_model, v_model, u_obs, v_obs, N_obs):
    """
    Calculates statistical error metrics between modelled and observed velocity fields:
    Root Mean Square Error (RMSE) and Mean Bias (MB) for u, v, and total velocity.

    Parameters:
    -----------
    u_model, v_model : ndarray
        Modelled east and north velocity components
    u_obs, v_obs : ndarray
        Observed east and north velocity components
    N_obs : int
        Number of valid observations used for normalisation

    Returns:
    --------
    tuple
        u_rms, v_rms      : RMSE for the u and v components
        u_mb, v_mb        : Mean bias for the u and v components
        total_rms         : Total RMSE (combined u and v)
        total_mb          : Total mean bias
    """
    # Temporal mean of observations and model
    u_mean_obs   = np.nanmean(u_obs)
    v_mean_obs   = np.nanmean(v_obs)
    u_model_mean = np.nanmean(u_model)
    v_model_mean = np.nanmean(v_model)

    # RMSE for each component
    u_rms = np.sqrt(np.nansum((u_obs - u_model)**2) / N_obs)
    v_rms = np.sqrt(np.nansum((v_obs - v_model)**2) / N_obs)

    # Mean bias: difference of anomalies (observation anomaly minus model anomaly)
    u_mb = np.nansum((u_obs - u_mean_obs) - (u_model - u_model_mean)) / N_obs
    v_mb = np.nansum((v_obs - v_mean_obs) - (v_model - v_model_mean)) / N_obs

    # Total (vector) RMSE and mean bias
    total_rms = np.sqrt(np.nansum((u_obs - u_model)**2 + (v_obs - v_model)**2) / N_obs)
    total_mb  = np.nansum(
        (u_obs - u_mean_obs) - (u_model - u_model_mean) +
        (v_obs - v_mean_obs) - (v_model - v_model_mean)
    ) / N_obs

    return u_rms, v_rms, u_mb, v_mb, total_rms, total_mb


# --------------------------------------------------------------


def var_dev(data):
    """
    Calculates the sample variance and standard deviation of a 1D dataset.

    Parameters:
    -----------
    data : array-like
        Input data values (NaNs are ignored in the mean calculation)

    Returns:
    --------
    var : float
        Sample variance (divided by n-1)
    std_dev : float
        Sample standard deviation
    """
    n = len(data)
    data_mean = np.nanmean(data)

    # Sum of squared deviations from the mean
    squared_deviations = [(x - data_mean) ** 2 for x in data]
    sum_sq = np.nansum(squared_deviations)

    # Sample variance (Bessel's correction: divide by n-1)
    var = sum_sq / (n - 1)

    # Sample standard deviation
    std_dev = math.sqrt(var)

    return var, std_dev


# --------------------------------------------------------------


def correlation(u_interp, v_interp, u_model, v_model):
    """
    Calculates Pearson correlation coefficients between interpolated and modelled
    velocity components (u, v) and their magnitudes.

    Parameters:
    -----------
    u_interp, v_interp : array-like
        Interpolated east and north velocity components (e.g. from observations)
    u_model, v_model : array-like
        Modelled east and north velocity components

    Returns:
    --------
    u_corr, v_corr, mag_corr : float
        Pearson correlation coefficient for u, v, and velocity magnitude
    """
    # Build masks to exclude NaN values
    valid_u_mask   = ~np.isnan(u_interp) & ~np.isnan(u_model)
    valid_v_mask   = ~np.isnan(v_interp) & ~np.isnan(v_model)
    valid_mag_mask = (~np.isnan(u_interp) & ~np.isnan(v_interp) &
                      ~np.isnan(u_model)  & ~np.isnan(v_model))

    # Pearson correlation for u and v individually
    u_corr = np.corrcoef(u_interp[valid_u_mask], u_model[valid_u_mask])[0, 1]
    v_corr = np.corrcoef(v_interp[valid_v_mask], v_model[valid_v_mask])[0, 1]

    # Pearson correlation for velocity magnitude
    mag_interp = np.sqrt(u_interp**2 + v_interp**2)
    mag_model  = np.sqrt(u_model**2 + v_model**2)

    mag_corr = np.corrcoef(mag_interp[valid_mag_mask], mag_model[valid_mag_mask])[0, 1]

    print(f"U correlation: {u_corr:.3f}")
    print(f"V correlation: {v_corr:.3f}")
    print(f"Total velocity correlation: {mag_corr:.3f}")

    return u_corr, v_corr, mag_corr


# --------------------------------------------------------------


def vorticity(u, v, lon, lat, nx=None, ny=None):
    """
    Calculates the vertical component of relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    using centred finite differences on a spherical Earth.
    Accepts both 2D arrays and 1D flattened vectors.

    Parameters:
    -----------
    u, v : ndarray
        East and north velocity components (2D or 1D)
    lon, lat : ndarray
        Longitude and latitude grids (2D or 1D, matching u and v)
    nx : int, optional
        Number of grid points in the x-direction (required for 1D input)
    ny : int, optional
        Number of grid points in the y-direction (required for 1D input)

    Returns:
    --------
    vorticity : ndarray
        Vorticity field (s⁻¹); same shape as input (returns 1D if input was 1D)
    """
    # If inputs are 1D, reshape to 2D for finite-difference calculations
    if u.ndim == 1:
        if nx is None or ny is None:
            raise ValueError("For 1D vectors, please provide grid dimensions nx and ny")

        u   = u.reshape(ny, nx)
        v   = v.reshape(ny, nx)
        lon = lon.reshape(ny, nx)
        lat = lat.reshape(ny, nx)

        return_1d = True
    else:
        return_1d = False

    ny, nx = u.shape
    vorticity = np.full_like(u, np.nan)

    R = 6.371e6  # Earth's radius in metres

    print(nx, ny)

    # Central differences on interior grid points (skip boundary rows/columns)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # Skip grid points with any NaN neighbour
            if (np.isnan(u[i, j]) or np.isnan(v[i, j]) or
                    np.isnan(u[i+1, j]) or np.isnan(u[i-1, j]) or
                    np.isnan(v[i, j+1]) or np.isnan(v[i, j-1])):
                continue

            # Grid spacing in metres (spherical Earth)
            dx = R * np.cos(lat[i, j] * np.pi / 180) * (lon[i, j+1] - lon[i, j-1]) * np.pi / 180
            dy = R * (lat[i+1, j] - lat[i-1, j]) * np.pi / 180

            # Centred finite differences: ∂v/∂x and ∂u/∂y
            dvdx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
            dudy = (u[i+1, j] - u[i-1, j]) / (2 * dy)

            print("i = ", i, " j = ", j)
            print("lat = ", lat[i, j])
            print("lon[i, j+1] = ", lon[i, j+1], " lon[i, j-1] = ", lon[i, j-1])
            print("lat[i+1, j] = ", lat[i+1, j], " lat[i-1, j] = ", lat[i-1, j])
            print("u[i+1, j] = ", u[i+1, j], " u[i-1, j] = ", u[i-1, j])
            print("v[i, j+1] = ", v[i, j+1], " v[i, j-1] = ", v[i, j-1])

            # Relative vorticity: ζ = ∂v/∂x - ∂u/∂y
            vorticity[i, j] = dvdx - dudy

    # Return in the same format as the input
    return vorticity.flatten() if return_1d else vorticity


# --------------------------------------------------------------


def divergence(u, v, lon, lat, nx=None, ny=None):
    """
    Calculates the horizontal divergence (∇·u = ∂u/∂x + ∂v/∂y) using centred
    finite differences on a spherical Earth.
    Accepts both 2D arrays and 1D flattened vectors.

    Parameters:
    -----------
    u, v : ndarray
        East and north velocity components (2D or 1D)
    lon, lat : ndarray
        Longitude and latitude grids (2D or 1D, matching u and v)
    nx : int, optional
        Number of grid points in the x-direction (required for 1D input)
    ny : int, optional
        Number of grid points in the y-direction (required for 1D input)

    Returns:
    --------
    divergence : ndarray
        Divergence field (s⁻¹); same shape as input (returns 1D if input was 1D)
    """
    # If inputs are 1D, reshape to 2D for finite-difference calculations
    if u.ndim == 1:
        if nx is None or ny is None:
            raise ValueError("For 1D vectors, please provide grid dimensions nx and ny")

        u   = u.reshape(ny, nx)
        v   = v.reshape(ny, nx)
        lon = lon.reshape(ny, nx)
        lat = lat.reshape(ny, nx)

        return_1d = True
    else:
        return_1d = False

    ny, nx = u.shape
    divergence = np.full_like(u, np.nan)

    R = 6.371e6  # Earth's radius in metres

    # Central differences on interior grid points (skip boundary rows/columns)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # Skip grid points with any NaN neighbour
            if (np.isnan(u[i, j]) or np.isnan(v[i, j]) or
                    np.isnan(u[i+1, j]) or np.isnan(u[i-1, j]) or
                    np.isnan(v[i, j+1]) or np.isnan(v[i, j-1])):
                continue

            # Grid spacing in metres (spherical Earth)
            dx = R * np.cos(lat[i, j] * np.pi / 180) * (lon[i, j+1] - lon[i, j-1]) * np.pi / 180
            dy = R * (lat[i+1, j] - lat[i-1, j]) * np.pi / 180

            # Centred finite differences: ∂u/∂x and ∂v/∂y
            dudx = (u[i, j+1] - u[i, j-1]) / (2 * dx)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2 * dy)

            # Horizontal divergence: ∇·u = ∂u/∂x + ∂v/∂y
            divergence[i, j] = dudx + dvdy

    # Return in the same format as the input
    return divergence.flatten() if return_1d else divergence


# --------------------------------------------------------------


def kinetic_energy(u, v):
    """
    Calculates the kinetic energy per unit mass: KE = 0.5 * (u² + v²).

    Parameters:
    -----------
    u : ndarray
        East velocity component (m/s)
    v : ndarray
        North velocity component (m/s)

    Returns:
    --------
    ke : ndarray
        Kinetic energy field (m²/s²)
    """
    return 0.5 * (u**2 + v**2)
