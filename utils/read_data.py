import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc

import pandas as pd
import os
from datetime import datetime, timedelta
import re

# -----------------------------------------------------------------------
# Module: read_data.py
# Purpose: I/O utilities for reading oceanographic data from NetCDF,
#          CSV, and proprietary text formats (RUV, TUV, TOTL).
#          Covers SSH (sea surface height), velocity fields, bathymetry,
#          radar observations, and DIVAnd output files.
# -----------------------------------------------------------------------


def read_nc_eta(filename, lon_name, lat_name, eta_name, time_snapshot, depth_index=None):
    """
    Reads a NetCDF file and extracts longitude, latitude, and sea surface
    height (SSH / eta) values.

    Parameters:
    -----------
    filename : str
        Path to the NetCDF file
    lon_name : str
        Name of the longitude variable in the file
    lat_name : str
        Name of the latitude variable in the file
    eta_name : str
        Name of the SSH variable in the file
    time_snapshot : int or str
        Time index or slice string (e.g. '0', '409:424', ':424', '409:')
    depth_index : int, optional
        Depth level index; if None, no depth dimension is assumed

    Returns:
    --------
    lon : ndarray
        Longitude values in radians
    lat : ndarray
        Latitude values in radians
    zeta : ndarray
        Sea surface height array
    """
    nc_file = Dataset(filename, mode='r')

    # Read coordinates and convert from degrees to radians
    lat = nc_file.variables[lat_name][:] * np.pi / 180
    lon = nc_file.variables[lon_name][:] * np.pi / 180

    # Parse the time_snapshot parameter into a slice or integer index
    if isinstance(time_snapshot, str):
        if ':' in time_snapshot:
            parts = time_snapshot.split(':')
            if time_snapshot.startswith(':'):       # Format ':424'
                time_slice = slice(0, int(parts[1]))
            elif time_snapshot.endswith(':'):       # Format '409:'
                time_slice = slice(int(parts[0]), None)
            else:                                   # Format '409:424'
                time_slice = slice(int(parts[0]), int(parts[1]))
        else:
            # Single index as a string
            time_slice = int(time_snapshot)
    else:
        # Single index as an integer
        time_slice = time_snapshot

    # Extract the SSH field at the specified time (and depth, if provided)
    if depth_index is not None:
        zeta = nc_file.variables[eta_name][time_slice, depth_index, :, :].squeeze()
    else:
        zeta = nc_file.variables[eta_name][time_slice, :, :].squeeze()

    nc_file.close()

    return lon, lat, zeta


def read_nc_vel(filename, lon_name, lat_name, u_name, v_name, time_snapshot,
                depth_index=None, lat_min_i=None, lat_max_i=None,
                lon_min_i=None, lon_max_i=None):
    """
    Reads a NetCDF file and extracts longitude, latitude, and the east (u)
    and north (v) velocity components.

    Supports optional spatial sub-setting by index range and depth selection.

    Parameters:
    -----------
    filename : str
        Path to the NetCDF file
    lon_name, lat_name : str
        Names of the longitude and latitude variables
    u_name, v_name : str
        Names of the east and north velocity variables
    time_snapshot : int or str
        Time index or slice string (e.g. '0', '0:24', ':24', '409:')
    depth_index : int, optional
        Depth level index; if None, no depth dimension is assumed
    lat_min_i, lat_max_i : int, optional
        Index bounds for latitude sub-setting
    lon_min_i, lon_max_i : int, optional
        Index bounds for longitude sub-setting

    Returns:
    --------
    lon, lat : ndarray
        Coordinate arrays in radians (sub-set if bounds provided)
    u, v : ndarray
        Velocity component arrays
    """
    nc_file = Dataset(filename, mode='r')

    # Read coordinates and convert from degrees to radians
    lat = nc_file.variables[lat_name][:] * np.pi / 180
    lon = nc_file.variables[lon_name][:] * np.pi / 180

    # Parse the time_snapshot parameter
    if isinstance(time_snapshot, str):
        if ':' in time_snapshot:
            parts = time_snapshot.split(':')
            if time_snapshot.startswith(':'):
                time_slice = slice(0, int(parts[1]))
            elif time_snapshot.endswith(':'):
                time_slice = slice(int(parts[0]), None)
            else:
                time_slice = slice(int(parts[0]), int(parts[1]))
        else:
            time_slice = int(time_snapshot)
    else:
        time_slice = time_snapshot

    # Build spatial index slices (full range if bounds are not specified)
    lat_slice = slice(lat_min_i, lat_max_i) if (lat_min_i is not None or lat_max_i is not None) else slice(None)
    lon_slice = slice(lon_min_i, lon_max_i) if (lon_min_i is not None or lon_max_i is not None) else slice(None)

    lon = lon[lon_slice]
    lat = lat[lat_slice]

    # Extract velocity fields
    if depth_index is not None:
        u = nc_file.variables[u_name][time_slice, depth_index, lat_slice, lon_slice]
        v = nc_file.variables[v_name][time_slice, depth_index, lat_slice, lon_slice]
    else:
        u = nc_file.variables[u_name][time_slice, lat_slice, lon_slice]
        v = nc_file.variables[v_name][time_slice, lat_slice, lon_slice]

    nc_file.close()

    return lon, lat, u, v


def catalan_sea_zone():
    """
    Reads a reference MITgcm NetCDF file to obtain the bounding-box
    coordinates (in radians) of the Catalan Sea simulation domain.

    Returns:
    --------
    lon_min, lon_max : float
        Western and eastern longitude limits (radians)
    lat_min, lat_max : float
        Southern and northern latitude limits (radians)
    """
    file = '/home/jgarcia/Projects/mar_catala/velocity_2D/january_2022/mitgcm_savitri/data/january/ALL_ave_files_Eta.nc'

    nc_file = Dataset(file, mode='r')
    lat_mit = nc_file.variables['latitude'][:] * np.pi / 180   # radians
    lon_mit = nc_file.variables['longitude'][:] * np.pi / 180  # radians
    nc_file.close()

    # Extract the domain limits from the MITgcm grid
    lon_min, lon_max = np.min(lon_mit), np.max(lon_mit)
    lat_min, lat_max = np.min(lat_mit), np.max(lat_mit)

    return lon_min, lon_max, lat_min, lat_max


def read_bath(file_bath):
    """
    Reads a bathymetry NetCDF file and returns longitude, latitude, and
    depth values. Fill values (1e+20) are replaced with NaN.

    Parameters:
    -----------
    file_bath : str
        Path to the bathymetry NetCDF file

    Returns:
    --------
    lon_bat, lat_bat : ndarray
        Longitude and latitude coordinate arrays
    h : ndarray
        Bathymetric depth array (NaN at fill-value points)
    """
    nc_bat = Dataset(file_bath, "r")

    lat_bat = nc_bat.variables['lat'][:]
    lon_bat = nc_bat.variables['lon'][:]
    h = nc_bat.variables["elevation"][:]

    nc_bat.close()

    # Replace fill values with NaN
    h[h == 1e+20] = np.nan

    return lon_bat, lat_bat, h


def valid_data_filter(lat, lon, zeta, lat_min, lat_max, lon_min, lon_max):
    """
    Filters SSH data to a geographic bounding box defined by index-space limits.

    Parameters:
    -----------
    lat, lon : ndarray (1D)
        Latitude and longitude arrays (radians)
    zeta : ndarray (2D or 3D)
        SSH field with shape [lat, lon] or [time, lat, lon]
    lat_min, lat_max : float
        Latitude bounds for filtering (same units as lat)
    lon_min, lon_max : float
        Longitude bounds for filtering (same units as lon)

    Returns:
    --------
    lon_filtered, lat_filtered : ndarray
        Sub-set coordinate arrays
    zeta_filtered_final : ndarray
        Filtered SSH array
    """
    # Boolean masks for coordinates within the bounding box
    lat_filter = (lat >= lat_min) & (lat <= lat_max)
    lon_filter = (lon >= lon_min) & (lon <= lon_max)

    lat_filtered = lat[lat_filter]
    lon_filtered = lon[lon_filter]

    # Apply filtering; handle both 2D [lat, lon] and 3D [time, lat, lon] arrays
    if len(zeta.shape) == 3:
        zeta_filtered_final = zeta[:, lat_filter][:, :, lon_filter]
    else:
        zeta_filtered_final = zeta[lat_filter][:, lon_filter]

    return lon_filtered, lat_filtered, zeta_filtered_final


def valid_data_filter_uv(lon, lat, u, v, mask, lat_min, lat_max, lon_min, lon_max):
    """
    Filters velocity (u, v) and mask data to a geographic bounding box.

    Parameters:
    -----------
    lon, lat : ndarray (1D)
        Longitude and latitude arrays (radians)
    u, v : ndarray (2D or 3D)
        East and north velocity fields with shape [lat, lon] or [time, lat, lon]
    mask : ndarray (2D)
        Land/sea mask with shape [lat, lon]
    lat_min, lat_max : float
        Latitude bounds for filtering
    lon_min, lon_max : float
        Longitude bounds for filtering

    Returns:
    --------
    lon_filtered, lat_filtered : ndarray
        Sub-set coordinate arrays
    u_filtered_final, v_filtered_final : ndarray
        Filtered velocity arrays
    mask_filtered : ndarray
        Filtered mask array
    """
    # Boolean masks for coordinates within the bounding box
    lat_filter = (lat >= lat_min) & (lat <= lat_max)
    lon_filter = (lon >= lon_min) & (lon <= lon_max)

    lat_filtered = lat[lat_filter]
    lon_filtered = lon[lon_filter]

    mask_filtered = mask[lat_filter][:, lon_filter]

    # Apply filtering; handle both 2D and 3D arrays
    if len(u.shape) == 3:
        u_filtered_final = u[:, lat_filter][:, :, lon_filter]
        v_filtered_final = v[:, lat_filter][:, :, lon_filter]
    else:
        u_filtered_final = u[lat_filter][:, lon_filter]
        v_filtered_final = v[lat_filter][:, lon_filter]

    return lon_filtered, lat_filtered, u_filtered_final, v_filtered_final, mask_filtered


def read_mask(filename, depth_index=None):
    """
    Reads an ocean/land mask from a NetCDF file (e.g. from a Copernicus
    model grid file).

    Parameters:
    -----------
    filename : str
        Path to the NetCDF mask file
    depth_index : int, optional
        Depth level index to extract; if None, the first level is used

    Returns:
    --------
    mask : ndarray (2D)
        Boolean or integer mask array
    """
    nc_mask = Dataset(filename, 'r')

    if depth_index is not None:
        mask = nc_mask.variables["mask"][depth_index, :, :]
    else:
        mask = nc_mask.variables["mask"][0, :, :]

    nc_mask.close()

    return mask


# -----------------------------------------------------------------------------------------------------------


def read_totl_files_date_range(base_directory, start_date, end_date):
    """
    Reads all TOTL_CATS files within a date range and organises the u and v
    velocity data into 3D matrices (time × longitude × latitude).

    Parameters:
    -----------
    base_directory : str
        Directory where the TOTL files are stored
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format or as a datetime object
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format or as a datetime object

    Returns:
    --------
    u_velocities : ndarray (n_days × n_lon × n_lat)
        East velocity component for each day
    v_velocities : ndarray (n_days × n_lon × n_lat)
        North velocity component for each day
    unique_longitudes : ndarray
        Sorted array of unique longitude values
    unique_latitudes : ndarray
        Sorted array of unique latitude values
    dates : list of datetime
        List of datetime objects, one per day in the range
    """
    # Convert string dates to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date")

    # Build a list of all dates in the range
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    n_days = len(dates)

    # Build the expected filename for each date
    files_dates = []
    for date in dates:
        filename = f"TOTL_CATS_{date.year}_{date.month:02d}_{date.day:02d}.tuv"
        files_dates.append((filename, date))

    def read_totl_file(file_path):
        """
        Reads a single TOTL TUV file and returns the parsed velocity data.
        Returns None if the file does not exist or cannot be parsed.
        """
        if not os.path.exists(file_path):
            return None

        data = []
        reading_data = False

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    if line == '%TableStart:':
                        reading_data = True
                        continue

                    if line == '%TableEnd:':
                        reading_data = False
                        break

                    # Parse data lines, skipping comment lines starting with '%%'
                    if reading_data and not line.startswith('%%'):
                        try:
                            values = line.split()
                            if len(values) >= 4:  # Minimum: lon, lat, u, v
                                lon = float(values[0])
                                lat = float(values[1])
                                u   = float(values[2])
                                v   = float(values[3])
                                data.append([lon, lat, u, v])
                        except (ValueError, IndexError):
                            continue  # Skip malformed lines

            return np.array(data) if data else None

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    # Read the first available file to determine the grid structure
    reference_data = None
    for filename, date in files_dates:
        file_path = os.path.join(base_directory, filename)
        temp_data = read_totl_file(file_path)
        if temp_data is not None:
            reference_data = temp_data
            break

    if reference_data is None:
        raise FileNotFoundError(f"No valid files found in directory: {base_directory}")

    # Extract unique coordinate values and determine the grid size
    unique_longitudes = np.unique(reference_data[:, 0])
    unique_latitudes  = np.unique(reference_data[:, 1])

    n_lon = len(unique_longitudes)
    n_lat = len(unique_latitudes)

    # Initialise output matrices with NaN
    u_velocities = np.full((n_days, n_lon, n_lat), np.nan)
    v_velocities = np.full((n_days, n_lon, n_lat), np.nan)

    # Build index look-up maps for fast coordinate matching
    lon_to_idx = {lon: i for i, lon in enumerate(unique_longitudes)}
    lat_to_idx = {lat: i for i, lat in enumerate(unique_latitudes)}

    files_processed = 0
    missing_files   = []

    # Read each file and populate the 3D velocity matrices
    for i, (filename, date) in enumerate(files_dates):
        file_path = os.path.join(base_directory, filename)
        data = read_totl_file(file_path)

        if data is not None:
            files_processed += 1

            for row in data:
                lon, lat, u, v = row

                # Find the closest longitude index (with floating-point tolerance)
                lon_idx      = None
                min_diff_lon = float('inf')
                for j, lon_ref in enumerate(unique_longitudes):
                    diff = abs(lon - lon_ref)
                    if diff < min_diff_lon:
                        min_diff_lon = diff
                        lon_idx = j
                        if diff < 1e-6:
                            break

                # Find the closest latitude index
                lat_idx      = None
                min_diff_lat = float('inf')
                for j, lat_ref in enumerate(unique_latitudes):
                    diff = abs(lat - lat_ref)
                    if diff < min_diff_lat:
                        min_diff_lat = diff
                        lat_idx = j
                        if diff < 1e-6:
                            break

                if lon_idx is not None and lat_idx is not None:
                    u_velocities[i, lon_idx, lat_idx] = u
                    v_velocities[i, lon_idx, lat_idx] = v
        else:
            missing_files.append(filename)

    return u_velocities, v_velocities, unique_longitudes, unique_latitudes, dates


def read_totl_files_month(base_directory, year, month):
    """
    Convenience wrapper that reads all TOTL files for a complete calendar month
    by delegating to read_totl_files_date_range.

    Parameters:
    -----------
    base_directory : str
        Directory where the TOTL files are stored
    year : int
        Target year
    month : int
        Target month (1–12)

    Returns:
    --------
    tuple
        (u_velocities, v_velocities, unique_longitudes, unique_latitudes, dates)
        as returned by read_totl_files_date_range
    """
    # Compute the first and last day of the month
    start_date = datetime(year, month, 1)

    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    return read_totl_files_date_range(base_directory, start_date, end_date)


def read_obs_csv(file_csv):
    """
    Reads a semicolon-separated CSV observation file containing HF radar
    radial velocity data.

    Parameters:
    -----------
    file_csv : str
        Path to the CSV file (no header; semicolon delimiter)

    Returns:
    --------
    obs_lon : ndarray
        Longitude of each observation (degrees)
    obs_lat : ndarray
        Latitude of each observation (degrees)
    u_rad : ndarray
        East velocity component (m/s; converted from cm/s)
    v_rad : ndarray
        North velocity component (m/s; converted from cm/s)
    angle_bearing : ndarray
        Bearing angle (degrees)
    r_vel : ndarray
        Radial velocity magnitude (m/s; converted from cm/s)
    angle_direction : ndarray
        Direction angle (degrees)
    flag_radar : ndarray
        Antenna/radar flag identifier
    """
    # Read the CSV into a DataFrame and convert to a NumPy matrix
    data_vel_csv = pd.read_csv(file_csv, sep=';', header=None)
    matrix_data = data_vel_csv.values

    # Extract individual observation arrays; convert velocities from cm/s to m/s
    obs_lon         = matrix_data[:, 0]
    obs_lat         = matrix_data[:, 1]
    u_rad           = matrix_data[:, 2] / 100   # cm/s → m/s
    v_rad           = matrix_data[:, 3] / 100   # cm/s → m/s
    angle_bearing   = matrix_data[:, 4]
    r_vel           = matrix_data[:, 5] / 100   # cm/s → m/s
    angle_direction = matrix_data[:, 6]
    flag_radar      = matrix_data[:, 7]

    return obs_lon, obs_lat, u_rad, v_rad, angle_bearing, r_vel, angle_direction, flag_radar


def grid_icatmar(file_icatmar):
    """
    Reads the ICATMAR (Institut Català de Recerca per a la Governança del Mar)
    grid from a NetCDF file and returns the arrays required by DIVAnd.

    Parameters:
    -----------
    file_icatmar : str
        Path to the ICATMAR NetCDF grid file

    Returns:
    --------
    mask : ndarray (bool, 2D)
        Ocean/land mask (True = ocean, i.e. elevation < 0)
    h_clean : ndarray (float32, 2D)
        Bathymetric depth array (NaN at land/missing points)
    pm, pn : ndarray (2D)
        Metric coefficients: reciprocal grid spacing in longitude and latitude
    lon_grid, lat_grid : ndarray (2D)
        Full 2D longitude and latitude grids ('ij' indexing)
    """
    ds = Dataset(file_icatmar, mode='r')

    # Read longitude, latitude, and elevation from the NetCDF file
    lon_ds = ds.variables['lon'][:]
    lat_ds = ds.variables['lat'][:]
    h      = ds.variables['elevation'][:, :]

    # Replace masked/fill values with NaN and cast to float32
    h_clean   = np.ma.filled(h,      np.nan).astype(np.float32)
    lon_clean = np.ma.filled(lon_ds, np.nan).astype(np.float32)
    lat_clean = np.ma.filled(lat_ds, np.nan).astype(np.float32)

    ds.close()

    # Ocean mask: grid points with elevation < 0 are ocean
    mask = (h_clean < 0).astype(bool)

    # Metric coefficients: reciprocal of the grid spacing (degrees)
    sz = (len(lon_clean), len(lat_clean))
    pm = np.ones(sz) / (lon_clean[1] - lon_clean[0])   # 1/Δlon
    pn = np.ones(sz) / (lat_clean[1] - lat_clean[0])   # 1/Δlat

    # Build full 2D coordinate grids ('ij' indexing matches Julia's meshgrid)
    lon_grid, lat_grid = np.meshgrid(lon_clean, lat_clean, indexing='ij')

    return mask, h_clean, pm, pn, lon_grid, lat_grid


def read_divand_file(filename):
    """
    Reads a DIVAnd output NetCDF file and extracts the reconstructed velocity
    fields (radial and total components) together with grid coordinates and time.

    Parameters:
    -----------
    filename : str
        Path to the DIVAnd NetCDF output file (.nc)

    Returns:
    --------
    lon : ndarray (2D)
        Longitude grid of the DIVAnd domain
    lat : ndarray (2D)
        Latitude grid of the DIVAnd domain
    time : ndarray (1D)
        Time vector
    uri : ndarray (3D, time × lat × lon)
        East (zonal) component of the radial velocity from DIVAnd
    vri : ndarray (3D, time × lat × lon)
        North (meridional) component of the radial velocity from DIVAnd
    uti : ndarray (3D, time × lat × lon)
        East (zonal) component of the total velocity from DIVAnd
    vti : ndarray (3D, time × lat × lon)
        North (meridional) component of the total velocity from DIVAnd
    """
    import netCDF4 as nc

    # Open the NetCDF file in read-only mode
    ds = nc.Dataset(filename)

    # Read the 2D spatial grid
    lon = ds["longitude"][:, :]   # Longitude of each grid node
    lat = ds["latitude"][:, :]    # Latitude of each grid node

    # Read the 1D time vector
    time = ds["time"][:]

    # Read the radial velocity components (3D: time × lat × lon)
    uri = ds["u_radial_divand"][:, :, :]   # East (zonal) radial component
    vri = ds["v_radial_divand"][:, :, :]   # North (meridional) radial component

    # Read the total velocity components (3D: time × lat × lon)
    uti = ds["u_total_divand"][:, :, :]    # East (zonal) total component
    vti = ds["v_total_divand"][:, :, :]    # North (meridional) total component

    # Close the file to free resources
    ds.close()

    return lon, lat, time, uri, vri, uti, vti
