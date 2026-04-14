from os import listdir
from os.path import isfile, join
import csv
import random
import os
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
import pandas as pd
from datetime import datetime
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import time
 
import re
from pathlib import Path
 
 
def latlon_to_m(lat1, lon1, lat2, lon2):
    """
    Calculates the distance in metres between two coordinates using the Haversine formula.
 
    Parameters:
    -----------
    lat1 : float
        Latitude of the first point (in degrees)
    lon1 : float
        Longitude of the first point (in degrees)
    lat2 : float
        Latitude of the second point (in degrees)
    lon2 : float
        Longitude of the second point (in degrees)
 
    Returns:
    --------
    float
        Distance in metres between the two points
    """
    R = 6371000  # Earth's radius in metres
 
    # Convert degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
 
    # Coordinate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
 
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
 
    distance = R * c
 
    return distance
 
 
def radars_cat():
    """
    Creates a dictionary assigning a numeric flag to each HF radar station
    along the Catalan Sea coast.
 
    Returns:
    --------
    dict
        Dictionary mapping radar station names to integer flags (1-indexed)
    """
    HFradar_name = ["AREN", "BEGU", "CREU", "GNST", "PBCN", "TOSS"]
    radar_dict = {name: i for i, name in enumerate(HFradar_name, start=1)}
    return radar_dict
 
 
def all_files_ruv(my_path):
    """
    Lists all RUV files found in a given directory.
 
    Parameters:
    -----------
    my_path : str
        Path to the directory to search
 
    Returns:
    --------
    list
        List of filenames with .ruv extension
    """
    onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f)) and f.lower().endswith('.ruv')]
    return onlyfiles
 
 
def finding_ruv_files(path_folder, year, month, day, hour):
    """
    Lists all RUV files in a folder matching a specific date and hour.
 
    Parameters:
    -----------
    path_folder : str
        Path to the directory containing RUV files
    year : int
        Year of the target timestamp
    month : int
        Month of the target timestamp
    day : int
        Day of the target timestamp
    hour : int
        Hour of the target timestamp
 
    Returns:
    --------
    list
        Sorted list of matching RUV filenames
    """
    # Format the date string to match the RUV filename convention
    full_date = f"{year}_{month:02d}_{day:02d}_{hour:02d}00"
 
    # RUV file pattern: [prefix]_[radar name]_YYYY_MM_DD_HHMM_[suffix].ruv
    ruv_pattern = f"*{full_date}*.ruv"
 
    # Search for matching files in the folder
    path_search = os.path.join(path_folder, ruv_pattern)
    ruv_files = glob.glob(path_search)
 
    ruv_filenames = [os.path.basename(file) for file in ruv_files]
 
    return sorted(ruv_filenames)
 
 
def read_ruv(my_path, list_ruv, radar_dict):
    """
    Reads all RUV files from a list and extracts velocity and geometry data.
 
    Parameters:
    -----------
    my_path : str
        Path to the directory containing the RUV files
    list_ruv : list
        List of RUV filenames to read
    radar_dict : dict
        Dictionary mapping radar names to antenna flags
 
    Returns:
    --------
    tuple
        lon, lat, u, v, angle_bearing, mod_vel, angle_direction, flag_antenna
        (all as lists)
    """
    lon = []
    lat = []
    u = []
    v = []
    mod_vel = []
    angle_bearing = []
    angle_direction = []
    flag_antenna = []
 
    for filename in list_ruv:
 
        file_ruv = my_path + filename
 
        # Extract the radar station name from the filename (characters 5–8)
        radar_name = filename[5:9]
 
        # Get the numeric flag for this radar station
        flag = radar_dict[radar_name]
 
        with open(file_ruv, 'r') as file:
            tot_lines = file.readlines()
            for line in tot_lines:
                line = line.strip()
                line = line.split()
 
                # Skip comment/header lines starting with '%'
                if line[0][0] == "%":
                    continue
 
                lon.append(line[0])
                lat.append(line[1])
                u.append(line[2])
                v.append(line[3])
                angle_bearing.append(line[16])   # Bearing angle
                mod_vel.append(line[17])          # Velocity magnitude
                angle_direction.append(line[18])  # Direction angle
 
                flag_antenna.append(flag)
 
    return lon, lat, u, v, angle_bearing, mod_vel, angle_direction, flag_antenna
 
 
def read_tuv(file_tuv):
    """
    Reads a TUV file (total velocity file from HF radar processing).
 
    Parameters:
    -----------
    file_tuv : str
        Path to the TUV file
 
    Returns:
    --------
    tuple
        lon, lat, u, v, mod_vel, angle, gdop  (all as lists)
        Returns longitude, latitude, velocity components, velocity magnitude,
        direction angle, and Geometric Dilution of Precision (GDOP).
    """
    lon = []
    lat = []
    u = []
    v = []
    mod_vel = []
    angle = []
    gdop = []
 
    with open(file_tuv, 'r') as file:
        tot_lines = file.readlines()
        for line in tot_lines:
            line = line.strip()
            line = line.split()
 
            # Skip comment/header lines starting with '%'
            if line[0][0] == "%":
                continue
 
            lon.append(line[0])
            lat.append(line[1])
            u.append(line[2])
            v.append(line[3])
            mod_vel.append(line[5])
            angle.append(line[6])
            gdop.append(line[10])
 
    return lon, lat, u, v, mod_vel, angle, gdop
 
 
def create_csv_radar(outfile, lon, lat, u, v, bearing, mod_vel, angle, flag_antenna):
    """
    Creates a CSV file with all radar data extracted from RUV files.
 
    Parameters:
    -----------
    outfile : str
        Path to the output CSV file
    lon : list
        Longitudes
    lat : list
        Latitudes
    u : list
        East velocity component
    v : list
        North velocity component
    bearing : list
        Bearing angles
    mod_vel : list
        Velocity magnitudes
    angle : list
        Direction angles
    flag_antenna : list
        Antenna flags
    """
    with open(outfile, 'w', newline='\n') as outfile:
        file = csv.writer(outfile, delimiter=';')
        for line in range(len(lon)):
            file.writerow([lon[line], lat[line], u[line], v[line],
                           bearing[line], mod_vel[line], angle[line], flag_antenna[line]])
    return
 
 
def create_all_csv_radar(path_ruv, radar_dict, year, month, ini_day, final_day, path_folder_csv):
    """
    Creates one CSV file per hourly snapshot over a given time range,
    iterating over days and hours.
 
    Parameters:
    -----------
    path_ruv : str
        Path to the folder containing RUV files
    radar_dict : dict
        Dictionary mapping radar names to antenna flags
    year : int
        Target year
    month : int
        Target month
    ini_day : int
        First day of the range (inclusive)
    final_day : int
        Last day of the range (exclusive)
    path_folder_csv : str
        Output folder for the generated CSV files
    """
    k = 0  # Global snapshot counter
 
    for i_day in range(ini_day, final_day):
        for i_hour in range(24):
 
            # List all RUV files for this day and hour
            f_ruv = finding_ruv_files(path_ruv, year, month, i_day, i_hour)
 
            print("DAY = ", i_day, " HOUR = ", i_hour)
            print(f_ruv)
 
            # Extract all relevant variables from the RUV files
            lon, lat, u, v, angle_bearing, mod_vel, angle_direction, flag_antenna = read_ruv(path_ruv, f_ruv, radar_dict)
 
            # Write the snapshot data to a CSV file
            outfile = path_folder_csv + "vel_radar_snapshot_" + f"{k:03d}" + ".csv"
            create_csv_radar(outfile, lon, lat, u, v, angle_bearing, mod_vel, angle_direction, flag_antenna)
 
            print("CSV FILE CREATED = ", "vel_radar_snapshot_" + f"{k:03d}" + ".csv")
 
            k = k + 1
 
    return
 
 
def read_obs_csv(file_csv):
    """
    Reads an observation CSV file and returns the relevant oceanographic variables.
 
    Parameters:
    -----------
    file_csv : str
        Path to the CSV file containing observation data
 
    Returns:
    --------
    tuple
        A tuple containing:
        - obs_lon     : longitude observations (degrees)
        - obs_lat     : latitude observations (degrees)
        - u_rad       : east velocity component (m/s)
        - v_rad       : north velocity component (m/s)
        - angle_bearing   : bearing angle (degrees)
        - r_vel       : radial velocity (m/s)
        - angle_direction : direction angle (degrees)
        - flag_radar  : radar/antenna flag
    """
    # Read the CSV file into a DataFrame (semicolon-separated, no header)
    data_vel_csv = pd.read_csv(file_csv, sep=';', header=None)
 
    # Convert to a NumPy matrix
    matrix_data = data_vel_csv.values
 
    # Extract individual observation arrays;
    obs_lon        = matrix_data[:, 0]
    obs_lat        = matrix_data[:, 1]
    u_rad          = matrix_data[:, 2]
    v_rad          = matrix_data[:, 3]
    angle_bearing  = matrix_data[:, 4]
    r_vel          = matrix_data[:, 5]
    angle_direction = matrix_data[:, 6]
    flag_radar     = matrix_data[:, 7]
 
    return obs_lon, obs_lat, u_rad, v_rad, angle_bearing, r_vel, angle_direction, flag_radar
 
 
def create_filtered_radials(obs_lon, obs_lat, u_radar, v_radar, angle_bearing, r_vel, angle_direction,
                            flag_radar, distance_max, output_file):
    """
    Filters radial observations by keeping only those that lie within a maximum
    distance threshold from a radial belonging to a different antenna.
    Results are written to an ASCII file.
 
    Parameters:
    -----------
    obs_lon : array-like
        Longitudes of the radial observations
    obs_lat : array-like
        Latitudes of the radial observations
    u_radar : array-like
        East velocity component of each radial
    v_radar : array-like
        North velocity component of each radial
    angle_bearing : array-like
        Bearing angles
    r_vel : array-like
        Radial velocity magnitudes
    angle_direction : array-like
        Direction angles
    flag_radar : array-like
        Antenna flags identifying the source radar
    distance_max : list of float
        List of maximum distance thresholds (in km) to test
    output_file : str
        Path to the output ASCII file
    """
    # Identify unique antenna flags
    antennas = np.unique(flag_radar)
    print("Detected antennas:", antennas)
 
    # Group radials by antenna flag
    radials_per_antenna = defaultdict(list)
    for i in range(len(obs_lon)):
        radials_per_antenna[flag_radar[i]].append(
            (obs_lon[i], obs_lat[i], u_radar[i], v_radar[i],
             angle_bearing[i], r_vel[i], angle_direction[i], flag_radar[i])
        )
 
    # Dictionary to store results for each distance threshold
    results = {}
 
    # Iterate over each maximum distance threshold
    for k in range(len(distance_max)):
 
        # Counter for filtered radials per antenna
        count_per_flag = {}
 
        # Lists to store filtered radials
        filtered_lon = []
        filtered_lat = []
        filtered_u = []
        filtered_v = []
        filtered_angle_bearing = []
        filtered_r_vel = []
        filtered_angle_direction = []
        filtered_flag = []
 
        # Compare radials from each antenna against all other antennas
        for antenna, radials in radials_per_antenna.items():
            for lon1, lat1, u1, v1, ang_bear1, vel1, ang_dir1, flag1 in radials:
                for other_antenna, other_radials in radials_per_antenna.items():
                    if other_antenna != antenna:  # Do not compare an antenna with itself
                        for lon2, lat2, *_ in other_radials:
                            if latlon_to_m(lat1, lon1, lat2, lon2) / 1000 < distance_max[k]:
                                # Store the radial observation that passed the filter
                                filtered_lon.append(lon1)
                                filtered_lat.append(lat1)
                                filtered_u.append(u1)
                                filtered_v.append(v1)
                                filtered_angle_bearing.append(ang_bear1)
                                filtered_r_vel.append(vel1)
                                filtered_angle_direction.append(ang_dir1)
                                filtered_flag.append(flag1)
 
                                # Increment radial count for this antenna
                                count_per_flag[flag1] = count_per_flag.get(flag1, 0) + 1
                                break  # Stop comparing once the condition is met
 
        print(f"\nMaximum distance: {distance_max[k]} km")
        print(f"Found {len(filtered_lon)} radials within {distance_max[k]} km of another antenna.")
        print("Filtered radial count per antenna:")
        for flag, count in count_per_flag.items():
            print(f"  Antenna {flag}: {count} radials")
 
        # Stack filtered data into a single matrix
        data = np.column_stack((
            filtered_lon, filtered_lat, filtered_u, filtered_v,
            filtered_angle_bearing, filtered_r_vel,
            filtered_angle_direction, filtered_flag
        ))
 
        # Write the filtered data to the output ASCII file
        np.savetxt(output_file, data, delimiter=' ')
        print(f"Data saved to file: {output_file}")
 
    return
 
 
def tuv_file_holes(input_file: str, output_file: str, percentage: float) -> None:
    """
    Reads a TUV file (radar total velocities), randomly removes a given percentage
    of data points, and writes the result to a new file with the same structure.
 
    Parameters:
    -----------
    input_file : str
        Path to the input TUV file
    output_file : str
        Path to the output TUV file (with holes)
    percentage : float
        Fraction of data points to remove (between 0 and 1)
    """
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")
 
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
 
    # Identify file sections: header, data, and footer
    header_lines = []
    data_lines = []
    footer_lines = []
 
    in_data_section = False
    data_start_idx = -1
    data_end_idx = -1
 
    for i, line in enumerate(lines):
        if line.strip() == '%TableStart:':
            in_data_section = True
            data_start_idx = i + 1
            header_lines = lines[:i+1]
            continue
        elif line.strip() == '%TableEnd:':
            in_data_section = False
            data_end_idx = i
            footer_lines = lines[i:]
            break
        elif in_data_section and not line.startswith('%%'):
            # Collect data lines (excluding comment lines within the data section)
            data_lines.append((i, line.strip()))
 
    # Calculate the number of points to remove
    N_holes = round(len(data_lines) * percentage)
 
    # Randomly select indices of data points to remove
    indices_to_remove = random.sample(range(len(data_lines)), N_holes)
    indices_to_remove.sort(reverse=True)  # Sort descending to avoid index shifting
 
    removed_data = []
 
    # Remove selected data points
    for idx in indices_to_remove:
        line_data = data_lines[idx][1]
        fields = line_data.split()
 
        if len(fields) >= 4:  # Ensure the line has enough fields
            lon = float(fields[0])
            lat = float(fields[1])
            u = float(fields[2])
            v = float(fields[3])
 
            removed_data.append((idx, lon, lat, u, v))
 
        del data_lines[idx]
 
    # Update the row count in the header
    updated_header = []
    for line in header_lines:
        if line.startswith('%TableRows:'):
            new_rows = len(data_lines)
            updated_header.append(f'%TableRows: {new_rows}\n')
        else:
            updated_header.append(line)
 
    # Write the modified file
    with open(output_file, 'w', encoding='utf-8') as f:
 
        f.writelines(updated_header)
 
        # Preserve in-section comment lines (lines starting with '%%')
        comment_lines = []
        for line in lines[data_start_idx:]:
            if line.startswith('%%'):
                comment_lines.append(line)
            else:
                break
        f.writelines(comment_lines)
 
        # Write the remaining data lines
        for _, data_line in data_lines:
            f.write(data_line + '\n')
 
        f.writelines(footer_lines)
 
    return
 
 
class HFRadarProcessor:
    """
    Python class to reconstruct total (u, v) velocity fields from HF radar
    radial velocity observations using the Least Squares combination method.
    """
 
    def parse_text_file(self, filepath: str) -> Dict:
        """
        Parses a text file containing radial velocity data.
        Expected columns:
            0: longitude
            1: latitude
            2: u (east component)
            3: v (north component)
            4: bearing angle
            5: velocity magnitude
            6: direction angle
            7: antenna flag (1–5)
 
        Parameters:
        -----------
        filepath : str
            Path to the input radial data file
 
        Returns:
        --------
        dict
            Dictionary with keys 'metadata', 'radials' (DataFrame), and 'site_info'
        """
        data = {
            'metadata': {},
            'radials': pd.DataFrame(),
            'site_info': {}
        }
 
        try:
            # Read the file as whitespace-separated plain text
            df = pd.read_csv(filepath, sep=r'\s+', header=None,
                             names=['LOND', 'LATD', 'VELU', 'VELV', 'BEAR', 'VELO', 'DIR', 'ANTENNA_FLAG'])
 
            print(f"File {os.path.basename(filepath)}: {len(df)} vectors read")
 
            if not df.empty:
 
                data['radials'] = df.copy()
 
                # Convert bearing to radians for subsequent calculations
                data['radials']['BEAR_RAD'] = np.radians(df['DIR'])
 
                # Build site information based on the antenna flag
                unique_flags = df['ANTENNA_FLAG'].unique()
 
                sites_info = {}
                for flag in unique_flags:
                    antenna_data = df[df['ANTENNA_FLAG'] == flag]
                    if not antenna_data.empty:
                        # Use the centroid of each antenna's data as an approximate position
                        mean_lat = antenna_data['LATD'].mean()
                        mean_lon = antenna_data['LOND'].mean()
 
                        sites_info[f'ANTENNA_{int(flag)}'] = {
                            'name': f'ANTENNA_{int(flag)}',
                            'latitude': mean_lat,
                            'longitude': mean_lon,
                            'flag': int(flag)
                        }
 
                data['site_info'] = sites_info
 
                print(f"  Detected antennas: {list(sites_info.keys())}")
                print(f"  Valid radial vectors: {len(data['radials'])}")
 
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
 
        return data
 
    def read_grid_from_netcdf(self, grid_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads a regular grid from a NetCDF file.
 
        Parameters:
        -----------
        grid_file : str
            Path to the NetCDF file containing the grid
 
        Returns:
        --------
        tuple
            LON, LAT (1D arrays), lon_grid, lat_grid (2D meshgrids), mask_antenna (3D array)
        """
        try:
            from netCDF4 import Dataset
 
            print(f"Reading grid from: {grid_file}")
 
            with Dataset(grid_file, 'r') as nc:
                LON = nc['lon'][:].astype(np.float64)
                LAT = nc['lat'][:].astype(np.float64)
                mask_antenna = nc['mask'][:, :, :]
 
                # Build 2D meshgrid
                lon_grid, lat_grid = np.meshgrid(LON, LAT)
 
                print(f"Latitude range: {LAT.min():.3f} - {LAT.max():.3f}")
                print(f"Longitude range: {LON.min():.3f} - {LON.max():.3f}")
 
                return LON, LAT, lon_grid, lat_grid, mask_antenna
 
        except FileNotFoundError:
            print(f"Error: File not found {grid_file}")
            print("Creating default grid...")
            return self.create_default_grid()
        except Exception as e:
            print(f"Error reading NetCDF file: {e}")
            print("Creating default grid...")
            return self.create_default_grid()
 
    def create_default_grid(self, lat_min: float = 40.0, lat_max: float = 43.0,
                            lon_min: float = 0.5, lon_max: float = 4.0,
                            grid_spacing: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a default regular grid if the NetCDF file cannot be read.
 
        Parameters:
        -----------
        lat_min, lat_max : float
            Latitude bounds of the grid
        lon_min, lon_max : float
            Longitude bounds of the grid
        grid_spacing : float
            Grid resolution in kilometres
 
        Returns:
        --------
        tuple
            lon_grid, lat_grid (2D meshgrids)
        """
        print("Creating default grid...")
 
        # Convert spacing from km to degrees (approximate)
        lat_step = grid_spacing / 111.0  # ~111 km per degree of latitude
        lon_step = grid_spacing / (111.0 * np.cos(np.radians((lat_min + lat_max) / 2)))
 
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)
 
        lon_grid, lat_grid = np.meshgrid(lons, lats)
 
        print(f"Default grid: {lat_grid.shape[0]} x {lon_grid.shape[1]} points")
 
        return lon_grid, lat_grid
 
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the great-circle distance between two points in kilometres
        using the Haversine formula.
 
        Parameters:
        -----------
        lat1, lon1 : float
            Coordinates of the first point (degrees)
        lat2, lon2 : float
            Coordinates of the second point (degrees)
 
        Returns:
        --------
        float
            Distance in kilometres
        """
        R = 6371.0  # Earth's radius in km
 
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
 
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
 
        return R * c
 
    def least_squares_combination(self, radials_data: List[Dict], grid_lons: np.ndarray,
                                  grid_lats: np.ndarray, max_distance: float = 6.0,
                                  min_observations: int = 2) -> pd.DataFrame:
        """
        Combines radial velocities from multiple antennas into total (u, v) velocity
        vectors at each grid point using the Least Squares method.
 
        For each grid point, all radial observations within max_distance are collected.
        The system A·x = b is solved with A containing the projection coefficients
        (cosine and sine of the bearing angle) and b the observed radial velocities.
 
        Parameters:
        -----------
        radials_data : list of dict
            List of parsed radial data dictionaries (from parse_text_file)
        grid_lons : ndarray
            2D array of grid longitudes
        grid_lats : ndarray
            2D array of grid latitudes
        max_distance : float
            Maximum search radius in kilometres
        min_observations : int
            Minimum number of observations required to compute a total velocity
 
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: longitude, latitude, u_total, v_total,
            magnitude, direction, gdop, n_obs, n_sites
        """
        t_start = time.perf_counter()
        results = []
        total_points = grid_lats.shape[0] * grid_lons.shape[1]
        processed_points = 0
 
        print(f"Processing {total_points} grid points...")
 
        # Loop over each grid point
        for i in range(grid_lats.shape[0]):
            for j in range(grid_lons.shape[1]):
 
                grid_lat = grid_lats[i, j]
                grid_lon = grid_lons[i, j]
 
                processed_points += 1
 
                if processed_points % 1000 == 0:
                    print(f"  Processed {processed_points}/{total_points} points...")
                    t_end = time.perf_counter()
                    print("TIME PROCESSING 1000 POINTS = ... ", (t_end - t_start), "s")
 
                # Collect nearby observations for this grid point
                observations_matrix = []
                antennas_represented = set()  # Track which antennas contribute
 
                for site_data in radials_data:
                    if site_data['radials'].empty:
                        continue
 
                    radials = site_data['radials']
 
                    for _, radial in radials.iterrows():
                        radial_lat = radial['LATD']
                        radial_lon = radial['LOND']
 
                        # Compute distance from the grid point to this radial observation
                        distance = self.calculate_distance(grid_lat, grid_lon, radial_lat, radial_lon)
 
                        if distance <= max_distance:
                            velo = radial['VELO']
                            bear_rad = radial['BEAR_RAD']
                            antenna_flag = radial['ANTENNA_FLAG']
 
                            observations_matrix.append([velo, bear_rad, distance])
                            antennas_represented.add(int(antenna_flag))
 
                # Only compute total velocity if at least 2 different antennas contribute
                if len(antennas_represented) >= 2 and len(observations_matrix) >= min_observations:
                    observations_matrix = np.array(observations_matrix)
 
                    try:
                        radial_velocities = observations_matrix[:, 0]
                        bearings = observations_matrix[:, 1] * (180 / np.pi)
                        distances = observations_matrix[:, 2]
 
                        # Convert bearing to mathematical angle (measured from East, counter-clockwise)
                        alpha = np.mod(450.0 - bearings, 360.0) * (np.pi / 180)
 
                        # Design matrix: columns are [cos(alpha), sin(alpha)]
                        A = np.column_stack([np.cos(alpha), np.sin(alpha)])
 
                        # Observation vector
                        b = radial_velocities
 
                        # Normal equations: (A^T A) x = A^T b
                        ATA = np.matmul(A.transpose(), A)
                        ATb = np.matmul(A.transpose(), b)
 
                        # Solve only if the system matrix is invertible
                        if np.linalg.det(ATA) != 0:
 
                            C = np.linalg.inv(ATA)
 
                            velocity = np.matmul(C, ATb)
                            u_vel, v_vel = velocity
 
                            vel_magnitude = np.sqrt(u_vel**2 + v_vel**2)
 
                            # Oceanographic convention: direction the current flows towards
                            # 0° = North, 90° = East
                            vel_direction = np.degrees(np.arctan2(u_vel, v_vel))
                            if vel_direction < 0:
                                vel_direction += 360
 
                            # Geometric Dilution of Precision (GDOP)
                            gdop_i = math.sqrt(np.abs(C.trace()))
 
                            result = {
                                'longitude': grid_lon,
                                'latitude': grid_lat,
                                'u_total': u_vel,
                                'v_total': v_vel,
                                'magnitude': vel_magnitude,
                                'direction': vel_direction,
                                'gdop': gdop_i,
                                'n_obs': observations_matrix.shape[0],
                                'n_sites': len(antennas_represented)
                            }
 
                            results.append(result)
 
                    except np.linalg.LinAlgError as e:
                        print(f"Error in least squares computation: {e}")
                        continue
 
        print(f"Done. Computed {len(results)} valid total velocities")
 
        return pd.DataFrame(results)
 
    def write_results_txt(self, output_path: str, results_df: pd.DataFrame):
        """
        Writes the total velocity results to a formatted text file.
 
        Parameters:
        -----------
        output_path : str
            Path to the output text file
        results_df : pd.DataFrame
            DataFrame containing the computed total velocities
        """
        try:
            # Build file header
            header = [
                "# HF Radar total velocities computed with Least Squares",
                f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Total points: {len(results_df)}",
                "#",
                "# Columns:",
                "# longitude (degrees) - Grid point longitude",
                "# latitude (degrees)  - Grid point latitude",
                "# u_total (cm/s)      - East velocity component",
                "# v_total (cm/s)      - North velocity component",
                "# magnitude (cm/s)    - Velocity magnitude",
                "# direction (degrees) - Current direction (0°=N, 90°=E)",
                "# gdop               - Geometrical Dilution of Precision",
                "#",
                "longitude\tlatitude\tu_total\tv_total\tmagnitude\tdirection\tgdop"
            ]
 
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header lines
                for line in header:
                    f.write(line + '\n')
 
                # Write data rows
                for _, row in results_df.iterrows():
                    f.write(f"{row['longitude']:.6f}\t{row['latitude']:.6f}\t"
                            f"{row['u_total']:.3f}\t{row['v_total']:.3f}\t"
                            f"{row['magnitude']:.3f}\t{row['direction']:.1f}\t{row['gdop']:.1f}\n")
 
            print(f"Results written to: {output_path}")
 
        except Exception as e:
            print(f"Error writing results file: {e}")
            raise
 
    def process_text_files(self, input_file: str, output_file: str,
                           grid_file: str = None,
                           max_distance: float = 6.0):
        """
        Main processing pipeline: reads a radial data text file and produces
        a total velocity text file using the Least Squares combination method.
 
        Parameters:
        -----------
        input_file : str
            Path to the input radial data file
        output_file : str
            Path to the output total velocity file
        grid_file : str, optional
            Path to a NetCDF file defining the output grid; if None, a default grid is used
        max_distance : float
            Search radius in kilometres for combining radials
        """
        print(f"=== HF Radar Total Velocity Processor ===")
        print(f"Input file:  {input_file}")
        print(f"Output file: {output_file}")
        print(f"Interpolation radius: {max_distance} km")
        print()
 
        # Verify that the input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found {input_file}")
            return
 
        print(f"Processing file: {input_file}")
 
        # Parse the radial data file
        data = self.parse_text_file(input_file)
 
        if data['radials'].empty:
            print("Error: No valid data could be extracted from the file")
            return
 
        all_radials = [data]  # Wrap in a list for compatibility with the combination method
 
        print(f"\nFile successfully processed")
        total_radials = len(data['radials'])
        print(f"Total radial vectors: {total_radials}")
 
        # Read or create the output grid
        if grid_file and os.path.exists(grid_file):
            print(f"\nReading grid from: {grid_file}")
            LON, LAT, grid_lons, grid_lats, mask_antenna = self.read_grid_from_netcdf(grid_file)
        else:
            print("\nUsing default grid")
            grid_lons, grid_lats = self.create_default_grid()
 
        # Combine radials into total velocities using Least Squares
        print(f"\nApplying Least Squares method...")
        results_df = self.least_squares_combination(all_radials, grid_lons, grid_lats, max_distance)
 
        if results_df.empty:
            print("Error: Could not compute total velocities")
            return
 
        # Print summary statistics
        print(f"\n=== Velocity statistics ===")
        print(f"Points with computed velocity: {len(results_df)}")
        print(f"Mean velocity:   {results_df['magnitude'].mean():.2f} cm/s")
        print(f"Maximum velocity: {results_df['magnitude'].max():.2f} cm/s")
        print(f"Minimum velocity: {results_df['magnitude'].min():.2f} cm/s")
        print(f"Standard deviation: {results_df['magnitude'].std():.2f} cm/s")
        print(f"Average observations per point: {results_df['n_obs'].mean():.1f}")
        print(f"Average antennas per point: {results_df['n_sites'].mean():.1f}")
 
        # Write the results to the output file
        print(f"\nWriting results...")
        self.write_results_txt(output_file, results_df)
 
        print(f"\nProcessing completed successfully!")
