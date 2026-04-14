import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

import glob
import xarray as xr
from datetime import datetime
import re


def extract_date_from_filename(filename):
    """
    Extracts the date from a NetCDF filename.

    Expects filenames that start with a date in 'YYYYMMDD_...' format.

    Args:
        filename (str): Full path or base name of the file.

    Returns:
        datetime: Parsed date, or None if the pattern is not found.
    """
    # Match the 8-digit date prefix at the start of the base filename
    date_match = re.match(r'(\d{8})_', os.path.basename(filename))
    if date_match:
        date_str = date_match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None


def process_netcdf_files(folder_path, start_date_str='20220201', end_date_str='20220228',
                         output_type='daily_averages'):
    """
    Processes NetCDF files in a given folder and extracts ocean current data (lon, lat, uo, vo).

    Files are filtered by the date range encoded in their filenames. Two processing
    modes are supported: computing daily averages, or concatenating all time steps.

    Args:
        folder_path (str): Path to the folder containing NetCDF files.
        start_date_str (str): Start date in 'YYYYMMDD' format (inclusive).
        end_date_str (str): End date in 'YYYYMMDD' format (inclusive).
        output_type (str): Processing mode:
                           - 'daily_averages': Computes daily averages (original behaviour).
                           - 'all_data': Concatenates all time steps without averaging.

    Returns:
        dict: Dictionary with arrays for lon, lat, u_data, v_data, and temporal info,
              or None if no valid files are found.
    """
    # Parse the date strings into datetime objects for comparison
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # Search for all NetCDF files (*.nc) in the specified folder
    netcdf_files = glob.glob(os.path.join(folder_path, "*.nc"))
    if not netcdf_files:
        # Fallback: list all files if no .nc files are found
        netcdf_files = glob.glob(os.path.join(folder_path, "*"))

    print(f"Found {len(netcdf_files)} files in folder")

    # Filter files whose filename date falls within the requested date range
    files_in_period = []
    for file_path in netcdf_files:
        file_date = extract_date_from_filename(file_path)
        if file_date and start_date <= file_date <= end_date:
            files_in_period.append((file_path, file_date))

    print(f"Found {len(files_in_period)} files within the specified period")

    if not files_in_period:
        print("No files found to process in the specified period")
        return None

    # Sort files chronologically before processing
    files_in_period.sort(key=lambda x: x[1])

    # Open the first file to retrieve the spatial grid dimensions (lon, lat)
    first_file = files_in_period[0][0]
    try:
        with xr.open_dataset(first_file) as ds:
            lon = ds['lon'].values
            lat = ds['lat'].values
    except Exception as e:
        print(f"Error opening first file: {e}")
        print(f"File path: {first_file}")
        try:
            print(f"Available variables: {list(xr.open_dataset(first_file).variables)}")
        except:
            pass
        return None

    # Dispatch to the appropriate processing function based on output_type
    if output_type == 'daily_averages':
        return _process_daily_averages(files_in_period, lon, lat, start_date, end_date)
    elif output_type == 'all_data':
        return _process_all_data(files_in_period, lon, lat)
    else:
        print(f"Invalid output type: {output_type}")
        print("Use 'daily_averages' or 'all_data'")
        return None


def _process_daily_averages(files_in_period, lon, lat, start_date, end_date):
    """
    Processes NetCDF files by computing daily averages of u and v ocean velocities.

    For each calendar day covered by the file list, all time steps from that day
    are averaged together and stored as a single record.

    Args:
        files_in_period (list): List of (file_path, file_date) tuples, sorted by date.
        lon (np.ndarray): Longitude array from the first file.
        lat (np.ndarray): Latitude array from the first file.
        start_date (datetime): Start of the date range.
        end_date (datetime): End of the date range.

    Returns:
        dict: Dictionary with keys 'lon', 'lat', 'u_data', 'v_data', 'dates', 'output_type'.
    """
    # Pre-allocate arrays for the maximum possible number of daily records
    num_days = (end_date - start_date).days + 1
    u_month = np.zeros((num_days, len(lat), len(lon)))
    v_month = np.zeros((num_days, len(lat), len(lon)))

    dates = []           # Will store the datetime for each processed day
    current_day = None   # Tracks the calendar day currently being accumulated
    daily_uo_data = []   # Accumulates u-velocity snapshots for the current day
    daily_vo_data = []   # Accumulates v-velocity snapshots for the current day
    day_index = 0        # Index into the pre-allocated daily arrays

    for file_path, file_date in files_in_period:
        try:
            print(f"Processing file: {os.path.basename(file_path)}")

            day = file_date.day

            # When the calendar day changes, flush the accumulated data as a daily average
            if current_day is not None and current_day != day and daily_uo_data:
                daily_uo_avg = np.mean(np.array(daily_uo_data), axis=0)
                daily_vo_avg = np.mean(np.array(daily_vo_data), axis=0)

                # Store the daily averages at the correct index
                u_month[day_index] = daily_uo_avg
                v_month[day_index] = daily_vo_avg

                dates.append(datetime(file_date.year, file_date.month, current_day))

                day_index += 1
                daily_uo_data = []
                daily_vo_data = []

            # Open the file and read the ocean velocity variables
            with xr.open_dataset(file_path) as ds:
                # Verify that the expected variables exist before proceeding
                if 'uo' not in ds or 'vo' not in ds:
                    print(f"Variables 'uo' or 'vo' not found in {os.path.basename(file_path)}")
                    print(f"Available variables: {list(ds.variables)}")
                    continue

                uo = ds['uo'].values
                vo = ds['vo'].values

                # If the arrays have a temporal dimension, average it out first
                # so each file contributes a single (lat, lon) snapshot
                if len(uo.shape) > 2:  # Shape is (time, lat, lon)
                    uo_mean = np.mean(uo, axis=0)
                    vo_mean = np.mean(vo, axis=0)
                    daily_uo_data.append(uo_mean)
                    daily_vo_data.append(vo_mean)
                else:  # Shape is already (lat, lon)
                    daily_uo_data.append(uo)
                    daily_vo_data.append(vo)

                current_day = day

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue

    # Flush the last accumulated day after the loop ends
    if daily_uo_data:
        daily_uo_avg = np.mean(np.array(daily_uo_data), axis=0)
        daily_vo_avg = np.mean(np.array(daily_vo_data), axis=0)

        u_month[day_index] = daily_uo_avg
        v_month[day_index] = daily_vo_avg

        dates.append(datetime(file_date.year, file_date.month, current_day))

    # Trim the pre-allocated arrays to the actual number of days processed
    u_month = u_month[:len(dates)]
    v_month = v_month[:len(dates)]

    print(f"Processing complete. Daily averages computed for {len(dates)} days.")

    return {
        'lon': lon,
        'lat': lat,
        'u_data': u_month,
        'v_data': v_month,
        'dates': dates,
        'output_type': 'daily_averages'
    }


def _process_all_data(files_in_period, lon, lat):
    """
    Processes NetCDF files by concatenating every individual time step.

    Unlike the daily averages mode, no temporal reduction is applied. Each time
    step from each file is stored as a separate record in the output arrays.

    Args:
        files_in_period (list): List of (file_path, file_date) tuples, sorted by date.
        lon (np.ndarray): Longitude array from the first file.
        lat (np.ndarray): Latitude array from the first file.

    Returns:
        dict: Dictionary with keys 'lon', 'lat', 'u_data', 'v_data', 'times', 'output_type',
              or None if no data could be read.
    """
    all_uo_data = []  # Accumulates every u-velocity snapshot across all files
    all_vo_data = []  # Accumulates every v-velocity snapshot across all files
    all_times = []    # Corresponding timestamp or date for each snapshot

    for file_path, file_date in files_in_period:
        try:
            print(f"Processing file: {os.path.basename(file_path)}")

            with xr.open_dataset(file_path) as ds:
                # Verify that the expected variables exist before proceeding
                if 'uo' not in ds or 'vo' not in ds:
                    print(f"Variables 'uo' or 'vo' not found in {os.path.basename(file_path)}")
                    print(f"Available variables: {list(ds.variables)}")
                    continue

                uo = ds['uo'].values
                vo = ds['vo'].values

                if len(uo.shape) == 3:  # Shape is (time, lat, lon)
                    # Iterate over each time step and store it individually
                    for t in range(uo.shape[0]):
                        all_uo_data.append(uo[t])
                        all_vo_data.append(vo[t])
                        # Use the file's internal time coordinate when available;
                        # otherwise fall back to the date encoded in the filename
                        if 'time' in ds:
                            all_times.append(ds['time'].values[t])
                        else:
                            all_times.append(file_date)

                elif len(uo.shape) == 2:  # Shape is (lat, lon) — single snapshot
                    all_uo_data.append(uo)
                    all_vo_data.append(vo)
                    all_times.append(file_date)

                else:
                    print(f"Unsupported array dimensions in {os.path.basename(file_path)}: {uo.shape}")
                    continue

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue

    if not all_uo_data:
        print("No data could be processed")
        return None

    # Stack the list of 2-D snapshots into a single 3-D array (time, lat, lon)
    u_all = np.array(all_uo_data)
    v_all = np.array(all_vo_data)

    print(f"Processing complete. Data concatenated for {len(all_times)} time steps.")

    return {
        'lon': lon,
        'lat': lat,
        'u_data': u_all,
        'v_data': v_all,
        'times': all_times,
        'output_type': 'all_data'
    }


def save_results(results, output_file='monthly_data.nc'):
    """
    Saves processing results to a NetCDF file using xarray.

    The output structure depends on the processing mode stored in results['output_type']:
    - 'daily_averages': dimensions are (day, lat, lon).
    - 'all_data':       dimensions are (time, lat, lon) with an integer time index.

    Args:
        results (dict): Dictionary returned by process_netcdf_files().
        output_file (str): Destination path for the output NetCDF file.
    """
    if results is None:
        print("No results to save")
        return

    if results['output_type'] == 'daily_averages':
        # Build the daily coordinate from the list of datetime objects
        days = np.array([d.day for d in results['dates']])

        ds = xr.Dataset(
            data_vars={
                'u_data': (['day', 'lat', 'lon'], results['u_data']),
                'v_data': (['day', 'lat', 'lon'], results['v_data'])
            },
            coords={
                'day': days,
                'lat': results['lat'],
                'lon': results['lon']
            },
            attrs={
                'description': 'Daily averages of ocean current velocities',
                'output_type': 'daily_averages',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )

    elif results['output_type'] == 'all_data':
        print("Saving all_data output...")

        print("u_data shape =", results['u_data'].shape)
        print("v_data shape =", results['v_data'].shape)

        # Use a simple integer index as the time coordinate
        time_indices = np.arange(len(results['times']))

        ds = xr.Dataset(
            data_vars={
                'u_data': (['time', 'lat', 'lon'], results['u_data']),
                'v_data': (['time', 'lat', 'lon'], results['v_data'])
            },
            coords={
                'time': time_indices,
                'lat': results['lat'],
                'lon': results['lon']
            },
            attrs={
                'description': 'All ocean current velocity time steps concatenated',
                'output_type': 'all_data',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        # Note: the full list of timestamps is available in results['times'] if needed;
        # it is not written to the file here to keep the attribute size manageable.

    # Write the dataset to disk in NetCDF4 format
    ds.to_netcdf(output_file)
    print(f"Results saved to {output_file}")

    return


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

# Input folder and date range configuration
folder_path = '../data/january_2026/data_medsea/'
start_date = '20260114'
end_date = '20260124'

# Output file path for the concatenated dataset
output_file_all = (
    '../data/gener_2026/data_medsea/all_data_january_2026.nc'
)

# Process all individual time steps without averaging
print(f"Processing NetCDF files from {start_date} to {end_date} — ALL DATA mode...")
results_all = process_netcdf_files(folder_path, start_date, end_date, output_type='all_data')

if results_all:
    print(f"Saving results to: {output_file_all}")
    save_results(results_all, output_file_all)

    # Print a summary of the resulting arrays
    print("\nSummary of results (all data):")
    print(f"  u_data shape:          {results_all['u_data'].shape}")
    print(f"  v_data shape:          {results_all['v_data'].shape}")
    print(f"  Longitude range:       [{results_all['lon'].min()}, {results_all['lon'].max()}]")
    print(f"  Latitude range:        [{results_all['lat'].min()}, {results_all['lat'].max()}]")
    print(f"  Total time steps:      {len(results_all['times'])}")
