import numpy as np
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import os
import netCDF4 as nc

import read_data as rd

# -----------------------------------------------------------------------
# Module: vel_eta.py
# Purpose: Computation of geostrophic velocities from sea surface height
#          (SSH / eta) gradients, along with utilities for reading
#          HF radar and Copernicus model data over time ranges, and
#          for projecting velocity fields onto the HF radar radial
#          geometry.
# -----------------------------------------------------------------------


def grid_icatmar(file_icatmar):
    """
    Reads the ICATMAR grid from a NetCDF file.

    Parameters:
    -----------
    file_icatmar : str
        Path to the NetCDF file

    Returns:
    --------
    mask : ndarray (bool, 2D)
        Ocean/land mask (True where elevation < 0, i.e. ocean)
    h_clean : ndarray (float32, 2D)
        Cleaned elevation array (NaN for missing/land values)
    pm, pn : ndarray (2D)
        Metric coefficients: reciprocal grid spacing in longitude and latitude
    lon_grid, lat_grid : ndarray (2D)
        Full 2D longitude and latitude grids ('ij' indexing)
    """
    with nc.Dataset(file_icatmar, "r") as ds:
        # Extract latitude, longitude, and elevation variables
        lon_ds = ds["lon"][:]
        lat_ds = ds["lat"][:]
        h      = ds["elevation"][:, :]

    # Replace masked/fill values with NaN and cast to float32
    h_clean   = np.where(np.ma.getmaskarray(h),      np.nan, h.data).astype(np.float32)
    lon_clean = np.where(np.ma.getmaskarray(lon_ds), np.nan, lon_ds.data).astype(np.float32)
    lat_clean = np.where(np.ma.getmaskarray(lat_ds), np.nan, lat_ds.data).astype(np.float32)

    # Ocean mask: True at grid points where elevation < 0
    mask = h_clean < 0

    # Metric coefficients: reciprocal of the grid spacing (degrees)
    sz = (lon_clean.shape[0], lat_clean.shape[0])
    pm = np.ones(sz) / (lon_clean[1] - lon_clean[0])   # 1/Δlon
    pn = np.ones(sz) / (lat_clean[1] - lat_clean[0])   # 1/Δlat

    # Build full 2D coordinate grids ('ij' indexing matches Julia's meshgrid convention)
    lon_grid, lat_grid = np.meshgrid(lon_clean, lat_clean, indexing="ij")

    return mask, h_clean, pm, pn, lon_grid, lat_grid


def constants():
    """
    Returns the main physical constants used in geostrophic calculations.

    Returns:
    --------
    Rt : float
        Earth's radius (m)
    g : float
        Gravitational acceleration (m/s²)
    omega : float
        Earth's rotation rate (rad/s)
    """
    Rt    = 6371e3      # Earth's radius [m]
    g     = 9.81        # Gravitational acceleration [m/s²]
    omega = 7.2921e-5   # Earth's angular velocity [rad/s]
    return Rt, g, omega


def coriolis_param(omega, fi):
    """
    Computes the Coriolis parameter f = 2·Ω·sin(φ).

    Parameters:
    -----------
    omega : float
        Earth's rotation rate (rad/s)
    fi : float or ndarray
        Latitude (radians)

    Returns:
    --------
    f : float or ndarray
        Coriolis parameter (s⁻¹)
    """
    return 2. * omega * np.sin(fi)


def grad_eta(Rt, avg_lat, eta, lon, lat):
    """
    Calculates the zonal (x) and meridional (y) gradients of the sea surface
    height (SSH / eta) using centred finite differences on a spherical Earth.
    Uses np.roll for periodic or near-periodic grids.

    Parameters:
    -----------
    Rt : float
        Earth's radius (m)
    avg_lat : float or ndarray
        Average latitude of the domain (radians); used for the cos(φ) correction
    eta : ndarray (2D or 3D)
        Sea surface height field (m); shape [lat, lon] or [time, lat, lon]
    lon : ndarray (2D)
        Longitude grid (radians)
    lat : ndarray (2D)
        Latitude grid (radians)

    Returns:
    --------
    gradient_eta_x : ndarray
        Zonal SSH gradient (∂η/∂x) in m/m (dimensionless)
    gradient_eta_y : ndarray
        Meridional SSH gradient (∂η/∂y) in m/m (dimensionless)
    """
    dim = len(np.shape(eta))

    if dim == 3:
        # 3D field [time, lat, lon]: roll along axes 2 (lon) and 1 (lat)
        gradient_eta_x = (np.roll(eta, -1, axis=2) - np.roll(eta, +1, axis=2)) / \
                         (np.roll(lon, -1, axis=1) - np.roll(lon, +1, axis=1)) / \
                         (Rt * np.cos(avg_lat))
        gradient_eta_y = (np.roll(eta, -1, axis=1) - np.roll(eta, +1, axis=1)) / \
                         (np.roll(lat, -1, axis=0) - np.roll(lat, +1, axis=0)) / Rt

    if dim == 2:
        # 2D field [lat, lon]: roll along axes 1 (lon) and 0 (lat)
        gradient_eta_x = (np.roll(eta, -1, axis=1) - np.roll(eta, +1, axis=1)) / \
                         (np.roll(lon, -1, axis=1) - np.roll(lon, +1, axis=1)) / \
                         (Rt * np.cos(avg_lat))
        gradient_eta_y = (np.roll(eta, -1, axis=0) - np.roll(eta, +1, axis=0)) / \
                         (np.roll(lat, -1, axis=0) - np.roll(lat, +1, axis=0)) / Rt

    return gradient_eta_x, gradient_eta_y


def vel_eta(g, f0, gradient_eta_x, gradient_eta_y):
    """
    Computes the geostrophic velocity components from SSH gradients using
    the geostrophic balance relations:
        u = -(g/f) · ∂η/∂y
        v =  (g/f) · ∂η/∂x

    Parameters:
    -----------
    g : float
        Gravitational acceleration (m/s²)
    f0 : float or ndarray
        Coriolis parameter (s⁻¹)
    gradient_eta_x : ndarray
        Zonal SSH gradient (∂η/∂x)
    gradient_eta_y : ndarray
        Meridional SSH gradient (∂η/∂y)

    Returns:
    --------
    u : ndarray
        Zonal (east) geostrophic velocity (m/s)
    v : ndarray
        Meridional (north) geostrophic velocity (m/s)
    """
    # Geostrophic balance constant C = g/f
    C = g / f0   # units: m/s² / s⁻¹ = m/s

    u = -C * gradient_eta_y   # Zonal velocity (m/s)
    v =  C * gradient_eta_x   # Meridional velocity (m/s)

    return u, v


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def month_puertos_radar(initial_time, final_time, base_path):
    """
    Reads hourly Puertos del Estado CODAR radar NetCDF files over a given time
    range and returns the velocity fields as 3D arrays (time × lat × lon).

    Parameters:
    -----------
    initial_time : str
        Start datetime in 'YYYY-MM-DD HH:MM' format
    final_time : str
        End datetime in 'YYYY-MM-DD HH:MM' format
    base_path : str
        Directory containing the CODAR NetCDF files

    Returns:
    --------
    lon_mesh_radar, lat_mesh_radar : ndarray (2D)
        2D longitude and latitude meshgrids
    all_u_radar : ndarray (n_hours × n_lat × n_lon)
        East velocity component for each hourly snapshot
    all_v_radar : ndarray (n_hours × n_lat × n_lon)
        North velocity component for each hourly snapshot
    """
    # Read a sample file to determine the grid dimensions
    sample_file = os.path.join(base_path, 'CODAR_EBRO_2022_01_01_0000.nc')
    lon_radar, lat_radar, _, _ = rd.read_nc_vel(sample_file, 'lon', 'lat', 'u', 'v', 0)

    date_ini = datetime.strptime(initial_time, "%Y-%m-%d %H:%M")
    date_end = datetime.strptime(final_time,   "%Y-%m-%d %H:%M")

    # Total number of hourly snapshots in the range (inclusive)
    n_hours = int(((date_end - date_ini).total_seconds() / 3600 + 1))

    n_lon = len(lon_radar)
    n_lat = len(lat_radar)
    all_u_radar = np.zeros((n_hours, n_lat, n_lon))
    all_v_radar = np.zeros((n_hours, n_lat, n_lon))

    hour_idx    = 0
    actual_date = date_ini

    # Loop over every hour in the time range
    while actual_date <= date_end:

        year     = actual_date.strftime("%Y")
        month    = actual_date.strftime("%m")
        day      = actual_date.strftime("%d")
        hour_str = actual_date.strftime("%H")

        filename  = f"CODAR_EBRO_{year}_{month}_{day}_{hour_str}00.nc"
        full_path = os.path.join(base_path, filename)

        _, _, u_radar, v_radar = rd.read_nc_vel(full_path, 'lon', 'lat', 'u', 'v', 0)

        # Replace fill values with NaN
        u_radar[u_radar == -9999.0] = np.nan
        v_radar[v_radar == -9999.0] = np.nan

        all_u_radar[hour_idx, :, :] = u_radar
        all_v_radar[hour_idx, :, :] = v_radar

        hour_idx    += 1
        actual_date += timedelta(hours=1)

    # Create 2D spatial coordinate meshgrids
    lon_mesh_radar, lat_mesh_radar = np.meshgrid(lon_radar, lat_radar)

    return lon_mesh_radar, lat_mesh_radar, all_u_radar, all_v_radar


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def avg_month_puertos_radar(initial_time, final_time, base_path):
    """
    Reads hourly Puertos del Estado CODAR radar NetCDF files and computes
    the daily average velocity for each day in the specified range.
    Results are stored as 3D arrays (n_days × lat × lon).

    Parameters:
    -----------
    initial_time : str
        Start datetime in 'YYYY-MM-DD HH:MM' format
    final_time : str
        End datetime in 'YYYY-MM-DD HH:MM' format
    base_path : str
        Directory containing the CODAR NetCDF files

    Returns:
    --------
    lon_mesh_radar, lat_mesh_radar : ndarray (2D)
        2D longitude and latitude meshgrids
    all_u_radar : ndarray (n_days × n_lat × n_lon)
        Daily-averaged east velocity component
    all_v_radar : ndarray (n_days × n_lat × n_lon)
        Daily-averaged north velocity component
    """
    # Read a sample file to determine the grid dimensions
    sample_file = os.path.join(base_path, 'CODAR_EBRO_2022_01_01_0000.nc')
    lon_radar, lat_radar, _, _ = rd.read_nc_vel(sample_file, 'lon', 'lat', 'u', 'v', 0)

    date_ini = datetime.strptime(initial_time, "%Y-%m-%d %H:%M")
    date_end = datetime.strptime(final_time,   "%Y-%m-%d %H:%M")

    n_days = (date_end - date_ini).days + 1
    n_lon  = len(lon_radar)
    n_lat  = len(lat_radar)

    all_u_radar = np.zeros((n_days, n_lat, n_lon))
    all_v_radar = np.zeros((n_days, n_lat, n_lon))

    for day_idx in range(n_days):

        current_date = date_ini + timedelta(days=day_idx)

        # Temporary arrays to accumulate the 24 hourly snapshots for this day
        day_u_data = np.zeros((24, n_lat, n_lon))
        day_v_data = np.zeros((24, n_lat, n_lon))

        for hour in range(24):

            actual_date = current_date + timedelta(hours=hour)

            year     = actual_date.strftime("%Y")
            month    = actual_date.strftime("%m")
            day      = actual_date.strftime("%d")
            hour_str = actual_date.strftime("%H")

            filename  = f"CODAR_EBRO_{year}_{month}_{day}_{hour_str}00.nc"
            full_path = os.path.join(base_path, filename)

            _, _, u_radar, v_radar = rd.read_nc_vel(full_path, 'lon', 'lat', 'u', 'v', 0)

            day_u_data[hour, :, :] = u_radar
            day_v_data[hour, :, :] = v_radar

            # Replace fill values with NaN before averaging
            day_u_data[day_u_data == -9999.0] = np.nan
            day_v_data[day_v_data == -9999.0] = np.nan

        # Compute the daily mean, ignoring NaN values
        all_u_radar[day_idx, :, :] = np.nanmean(day_u_data, axis=0)
        all_v_radar[day_idx, :, :] = np.nanmean(day_v_data, axis=0)

    # Create 2D spatial coordinate meshgrids
    lon_mesh_radar, lat_mesh_radar = np.meshgrid(lon_radar, lat_radar)

    return lon_mesh_radar, lat_mesh_radar, all_u_radar, all_v_radar


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def month_cop(initial_time, final_time, base_path):
    """
    Reads hourly Copernicus Mediterranean (CMCC MFSe3r1) model NetCDF files
    over a given date range and returns the velocity fields as 3D arrays
    (n_total_hours × lat × lon).

    Each daily file is assumed to contain 24 hourly snapshots.

    Parameters:
    -----------
    initial_time : str
        Start date in 'YYYY-MM-DD' format
    final_time : str
        End date in 'YYYY-MM-DD' format
    base_path : str
        Directory containing the Copernicus NetCDF files

    Returns:
    --------
    lon_mesh_model, lat_mesh_model : ndarray (2D)
        2D longitude and latitude meshgrids
    all_u_model : ndarray (n_hours × n_lat × n_lon)
        East velocity component for each hourly snapshot
    all_v_model : ndarray (n_hours × n_lat × n_lon)
        North velocity component for each hourly snapshot
    """
    # Read a sample file to determine the grid dimensions
    sample_file = os.path.join(base_path, '20220101_h-CMCC--RFVL-MFSe3r1-MED-b20240131_re-sv01.00.nc')
    lon_model, lat_model, _, _ = rd.read_nc_vel(sample_file, 'lon', 'lat', 'uo', 'vo', 0)

    date_ini = datetime.strptime(initial_time, "%Y-%m-%d")
    date_end = datetime.strptime(final_time,   "%Y-%m-%d")

    n_days  = (date_end - date_ini).days + 1
    n_hours = n_days * 24   # Total number of hourly snapshots

    n_lon = len(lon_model)
    n_lat = len(lat_model)
    all_u_model = np.zeros((n_hours, n_lat, n_lon))
    all_v_model = np.zeros((n_hours, n_lat, n_lon))

    hour_idx = 0

    for i_day in range(n_days):

        current_date = date_ini + timedelta(days=i_day)

        year  = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        day   = current_date.strftime("%d")

        filename  = f"{year}{month}{day}_h-CMCC--RFVL-MFSe3r1-MED-b20240131_re-sv01.00.nc"
        full_path = os.path.join(base_path, filename)

        # Read all 24 hourly snapshots from the daily file
        _, _, u_model, v_model = rd.read_nc_vel(full_path, 'lon', 'lat', 'uo', 'vo', '0:24')

        # Replace fill values (>= 1e20) with NaN
        u_model[u_model >= 1.0e+20] = np.nan
        v_model[v_model >= 1.0e+20] = np.nan

        # Store each hourly snapshot separately in the output array
        for hour in range(u_model.shape[0]):  # First dimension is the hour index
            all_u_model[hour_idx, :, :] = u_model[hour, :, :]
            all_v_model[hour_idx, :, :] = v_model[hour, :, :]
            hour_idx += 1

    # Create 2D spatial coordinate meshgrids
    lon_mesh_model, lat_mesh_model = np.meshgrid(lon_model, lat_model)

    return lon_mesh_model, lat_mesh_model, all_u_model, all_v_model


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def avg_month_cop(initial_time, final_time, base_path):
    """
    Reads hourly Copernicus Mediterranean model NetCDF files and computes the
    daily average velocity for each day in the specified date range.
    Results are stored as 3D arrays (n_days × lat × lon).

    Parameters:
    -----------
    initial_time : str
        Start date in 'YYYY-MM-DD' format
    final_time : str
        End date in 'YYYY-MM-DD' format
    base_path : str
        Directory containing the Copernicus NetCDF files

    Returns:
    --------
    lon_mesh_model, lat_mesh_model : ndarray (2D)
        2D longitude and latitude meshgrids
    all_u_model : ndarray (n_days × n_lat × n_lon)
        Daily-averaged east velocity component
    all_v_model : ndarray (n_days × n_lat × n_lon)
        Daily-averaged north velocity component
    """
    # Read a sample file to determine the grid dimensions
    sample_file = os.path.join(base_path, '20220101_h-CMCC--RFVL-MFSe3r1-MED-b20240131_re-sv01.00.nc')
    lon_model, lat_model, _, _ = rd.read_nc_vel(sample_file, 'lon', 'lat', 'uo', 'vo', 0)

    date_ini = datetime.strptime(initial_time, "%Y-%m-%d")
    date_end = datetime.strptime(final_time,   "%Y-%m-%d")

    n_days = (date_end - date_ini).days + 1
    n_lon  = len(lon_model)
    n_lat  = len(lat_model)

    all_u_model = np.zeros((n_days, n_lat, n_lon))
    all_v_model = np.zeros((n_days, n_lat, n_lon))

    for i_day in range(n_days):

        current_date = date_ini + timedelta(days=i_day)

        year  = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        day   = current_date.strftime("%d")

        filename  = f"{year}{month}{day}_h-CMCC--RFVL-MFSe3r1-MED-b20240131_re-sv01.00.nc"
        full_path = os.path.join(base_path, filename)

        # Read all 24 hourly snapshots and compute the daily mean
        _, _, u_model, v_model = rd.read_nc_vel(full_path, 'lon', 'lat', 'uo', 'vo', '0:24')

        all_u_model[i_day, :, :] = np.nanmean(u_model, axis=0)
        all_v_model[i_day, :, :] = np.nanmean(v_model, axis=0)

    # Create 2D spatial coordinate meshgrids
    lon_mesh_model, lat_mesh_model = np.meshgrid(lon_model, lat_model)

    return lon_mesh_model, lat_mesh_model, all_u_model, all_v_model


def radial_vel(angle_bearing, u_data, v_data):
    """
    Projects a total (u, v) velocity field onto the radial direction of each
    ICATMAR HF radar observation point to obtain synthetic radial velocities.

    The projection formula is:
        v_radial = u · sin(θ) + v · cos(θ)
    where θ is the bearing angle in radians.

    Parameters:
    -----------
    angle_bearing : ndarray
        Bearing angles at each observation point (degrees)
    u_data : ndarray
        East (zonal) velocity component at each observation point (m/s)
    v_data : ndarray
        North (meridional) velocity component at each observation point (m/s)

    Returns:
    --------
    u_rad : ndarray
        East component of the radial velocity vector (m/s)
    v_rad : ndarray
        North component of the radial velocity vector (m/s)
    vel_radial : ndarray
        Signed radial velocity magnitude (m/s)
    theta_rad : ndarray
        Bearing angles converted to radians
    """
    # Convert bearing angles from degrees to radians
    theta_rad = np.deg2rad(angle_bearing)

    # Project the total velocity onto the radial (bearing) direction
    vel_radial = u_data * np.sin(theta_rad) + v_data * np.cos(theta_rad)

    # Decompose the radial velocity back into east and north components
    u_rad = vel_radial * np.sin(theta_rad)
    v_rad = vel_radial * np.cos(theta_rad)

    return u_rad, v_rad, vel_radial, theta_rad


def putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                               u_array, v_array):
    """
    Assigns scattered data points to the nearest node of a Least Squares (LS)
    grid. Points that do not fall exactly on a grid node trigger a warning.

    Parameters:
    -----------
    lon_array : array-like
        Longitudes of the data points to assign
    lat_array : array-like
        Latitudes of the data points to assign
    lon_grid_LS : ndarray (2D)
        Longitude grid of the LS mesh
    lat_grid_LS : ndarray (2D)
        Latitude grid of the LS mesh
    u_array : array-like
        East velocity component at each data point
    v_array : array-like
        North velocity component at each data point

    Returns:
    --------
    u_LS_2d, v_LS_2d : ndarray (2D)
        Velocity components mapped onto the LS grid (NaN at unassigned nodes)
    """
    tolerance = 1e-6

    # Initialise output arrays with NaN
    u_LS_2d = np.full(lon_grid_LS.shape, np.nan)
    v_LS_2d = np.full(lon_grid_LS.shape, np.nan)

    points_found     = 0
    points_not_found = 0

    for j in range(len(lon_array)):

        # Find the closest grid node using Euclidean distance in degree space
        distances   = np.sqrt((lon_grid_LS - lon_array[j])**2 +
                              (lat_grid_LS - lat_array[j])**2)
        min_distance = np.min(distances)
        idx_closest  = np.unravel_index(np.argmin(distances), distances.shape)
        idx_lon, idx_lat = idx_closest

        # Check whether the closest node coincides with the data point
        lon_closest   = lon_grid_LS[idx_lon, idx_lat]
        lat_closest   = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance and
                         abs(lat_closest - lat_array[j]) < tolerance)

        if is_grid_point:
            points_found += 1
        else:
            points_not_found += 1

            if points_not_found <= 5:  # Show only the first 5 warnings
                print(f"WARNING: Point {j} is not exactly on the grid")
                print(f"  Distance:  {min_distance}")
                print(f"  Target:    ({lon_array[j]}, {lat_array[j]})")
                print(f"  Closest:   ({lon_closest}, {lat_closest})")

        # Assign the velocity values to the closest grid node
        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]

    return u_LS_2d, v_LS_2d
