"""
Script for creating synthetic radial velocity data from Copernicus model velocities.
Projects model total velocities onto radar beam directions to simulate radial observations.
"""

import numpy as np
from datetime import datetime, timedelta

from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main

Main.include("../utils/reading_obs.jl")

import sys
sys.path.append('../utils')

import read_data as rd
import vel_eta as ve 
import HFradar_data as hf
import mathematics as ma

# Initialize HF radar processor
processor = hf.HFRadarProcessor()

# CATALAN RADARS - Dictionary mapping radar stations to antenna flags
radar_dict = hf.radars_cat()

# ------------------------------------------------------------------------------------

# COPERNICUS GROUND TRUTH - Load model data for validation

# Input file: Copernicus model velocities (January 2026)
file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"

# Read Copernicus model velocities (u and v components)
# lon_cop, lat_cop are in radians, u_cop and v_cop are 3D arrays (time, lat, lon)
lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")

# Create 2D meshgrids from 1D coordinate arrays
lon_mesh_model_cop, lat_mesh_model_cop = np.meshgrid(lon_cop, lat_cop)

# Convert from radians to degrees for plotting and interpolation
lon_mesh_model_cop = lon_mesh_model_cop * (180 / np.pi)
lat_mesh_model_cop = lat_mesh_model_cop * (180 / np.pi)

# Replace fill values (1e20) with NaN for proper handling
u_model_cop = np.where(u_cop == 1.0e20, np.nan, u_cop).astype(np.float32)
v_model_cop = np.where(v_cop == 1.0e20, np.nan, v_cop).astype(np.float32)

print("COPERNICUS read")

# ------------------------------------------------------------------------------------

# Number of time steps to process (264 = 11 days × 24 hours)
n_times = 264

for i in range(1, n_times):
    
    print("ITERATION NUMBER = ... ", i-1)
    
    # Extract velocity field for the current time step
    u_cop_timestep = u_model_cop[i-1, :, :]
    v_cop_timestep = v_model_cop[i-1, :, :]
    
    # Radar observation file for this hour
    # Contains radial observations from real HF radars
    file_csv = "../data/january_2026/radials_10_days/csv_files_real_radar/vel_real_radar_snapshot_" + str(i-1).zfill(3) + ".csv"
    
    # Read radar observation data from CSV file
    # Returns: longitudes, latitudes, u_radar, v_radar, bearing_angle, radial_velocity, direction_angle, antenna_flag
    obs_lon, obs_lat, u_radar, v_radar, angle_bearing, vel_radar, angle_direction, flag_radar = Main.read_obs_csv(file_csv)
    
    # Interpolate Copernicus model velocities to radar observation points
    # This gives us the "true" total velocity at each radar measurement location
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop,
                                                     u_cop_timestep, v_cop_timestep,
                                                     obs_lon, obs_lat, points_mode=True)
    
    # Convert total velocities to radial velocities (project onto radar beam direction)
    # Uses the bearing angle of each radar to compute the radial component
    u_rad, v_rad, vel_radial, theta_rad = ve.radial_vel(angle_bearing, u_interp_cop, v_interp_cop)
    
    # Convert radial angle from radians to degrees
    theta_rad = theta_rad * (180 / np.pi)
    
    # Create mask to remove invalid points (NaN values in interpolated velocities)
    valid_mask = ~(np.isnan(u_rad) | np.isnan(v_rad))
    
    # Apply mask to filter only valid observations
    obs_lon = obs_lon[valid_mask]
    obs_lat = obs_lat[valid_mask]
    u_rad = u_rad[valid_mask]
    v_rad = v_rad[valid_mask]
    angle_bearing = angle_bearing[valid_mask]
    vel_radial = vel_radial[valid_mask]
    theta_rad = theta_rad[valid_mask]
    flag_radar = flag_radar[valid_mask]
    
    # Output file for synthetic radial velocities
    outfile_name = "../data/january_2026/radials_10_days/csv_files_radials_medsea/vel_radar_medsea_snapshot_" + str(i-1).zfill(3) + ".csv"
    
    # Save synthetic radial velocities to CSV file
    # Format matches real radar data for compatibility with downstream processing
    hf.create_csv_radar(outfile_name, obs_lon, obs_lat, u_rad, v_rad,
                        angle_bearing, vel_radial, theta_rad, flag_radar)
