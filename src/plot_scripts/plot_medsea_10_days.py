import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime, timedelta

# -----------------------------------------------------------------------
# Script: plot_medsea_10_days.py
# Purpose: Reads the Copernicus MEDSEA surface velocity field, interpolates
#          it to the HF radar LS grid (applying the combined antenna coverage
#          mask), and saves quiver velocity plots for each hourly snapshot.
#
# Pipeline:
#   1. Load the ICATMAR bathymetry and the LS grid.
#   2. Build the combined antenna coverage mask.
#   3. Loop over hourly snapshots:
#      a. Read the Copernicus velocity snapshot.
#      b. Interpolate to the LS grid and apply the coverage mask.
#      c. Save the quiver figure.
# -----------------------------------------------------------------------

import sys
sys.path.append('../../utils')

import read_data as rd
import fig_vel as fg
import HFradar_data as hf
import mathematics as ma

# Create an instance of HFRadarProcessor (used for grid reading)
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------

file_icatmar = '../../data/bathy.nc'
lon_bat, lat_bat, h = rd.read_bath(file_icatmar)

# ------------------------------------------------------------------------------------
# LS GRID AND ANTENNA COVERAGE MASK
# ------------------------------------------------------------------------------------

grid_file = "../../data/hfradar_totals_grid_icatmar.nc"
lon_LS, lat_LS, lon_grid, lat_grid, mask_antena = processor.read_grid_from_netcdf(grid_file)

# Build the full 2D LS coordinate meshgrid
lat_grid_LS, lon_grid_LS = np.meshgrid(lat_LS, lon_LS)

# Combined coverage mask for the BEGU, CREU, and TOSS radar group (indices 1–3)
mask_group1 = np.sum(mask_antena[1:4, :, :], axis=0)

# Combined coverage mask for the AREN, GNST, and PBCN radar group (indices 5–6)
mask_group2 = np.sum(mask_antena[5:7, :, :], axis=0)

# Total coverage mask: 1 where at least one radar covers the grid point
mask_total = mask_group1 + mask_group2
mask_total[mask_total >= 1] = 1

# Transpose and convert to float64; replace zero (uncovered) points with NaN
# so that they render as transparent in the quiver plot
mask_total_t = mask_total.T
mask_total_t_float = mask_total_t.astype(np.float64)
mask_total_t_float[mask_total_t_float == 0.0] = np.nan
mask_total_t = mask_total_t_float

print("SIZE MASK TOTAL =", mask_total_t.shape)

# ------------------------------------------------------------------------------------
# DOMAIN LIMITS (Catalan Sea, in radians)
# ------------------------------------------------------------------------------------

lat_min = 40.2 * (np.pi / 180)
lat_max = 42.7 * (np.pi / 180)
lon_min = 1.6  * (np.pi / 180)
lon_max = 4.0  * (np.pi / 180)

# ------------------------------------------------------------------------------------
# COPERNICUS LAND/SEA MASK
# ------------------------------------------------------------------------------------

file_mask = "../../data/MED-MFC_006_013_mask_bathy.nc"
mask_cop  = rd.read_mask(file_mask)

# ------------------------------------------------------------------------------------
# COPERNICUS MEDSEA SOURCE FILE AND PLOT PARAMETERS
# ------------------------------------------------------------------------------------

# file_cop = '../../data/february_2025/data_medsea/all_data_february_2025.nc'
file_cop = '../../data/january_2026/data_medsea/all_data_january_2026.nc'

pdt   = 1       # Arrow decimation factor
SCALE = 10      # Quiver arrow scale
WIDTH = 0.0025  # Quiver arrow width

# ------------------------------------------------------------------------------------
# MAIN LOOP: READ, INTERPOLATE, AND PLOT EACH HOURLY SNAPSHOT
# ------------------------------------------------------------------------------------

n_days      = 10
time_length = (n_days + 1) * 24   # Total number of hourly snapshots

for i in range(1):

    print("ITERATION = ... ", i)

    # Read the Copernicus velocity snapshot at time index i
    lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(
        file_cop, 'lon', 'lat', 'u_data', 'v_data', str(i)
    )

    # Replace fill values (1e20) with NaN
    u_model_cop = np.where(u_cop == 1.0e20, np.nan, u_cop)
    v_model_cop = np.where(v_cop == 1.0e20, np.nan, v_cop)

    # Build a 2D coordinate meshgrid and convert from radians to degrees
    lon_mesh_model_cop, lat_mesh_model_cop = np.meshgrid(lon_cop, lat_cop)
    lon_mesh_model_cop = lon_mesh_model_cop * (180 / np.pi)
    lat_mesh_model_cop = lat_mesh_model_cop * (180 / np.pi)

    # Interpolate the Copernicus field to the LS grid
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(
        lon_mesh_model_cop, lat_mesh_model_cop,
        u_model_cop, v_model_cop,
        lon_grid_LS, lat_grid_LS,
        mask_coarse_grid=[], points_mode=False
    )

    # Apply the antenna coverage mask: points outside radar coverage become NaN
    u_interp_cop_masked = u_interp_cop * mask_total_t
    v_interp_cop_masked = v_interp_cop * mask_total_t

    lon_mesh_model_cop_masked = lon_grid_LS * mask_total_t
    lat_mesh_model_cop_masked = lat_grid_LS * mask_total_t

    # Figure title and output path
    title_name = "Field velocity Copernicus - " + str(i) + " hours from 14/01/2026"
    fig_name   = '../../figures/january_2026/medsea_GT/GT_' + str(i).zfill(3) + '_h.png'

    # Generate and save the quiver velocity plot
    fg.vel_quiver(
        lon_mesh_model_cop_masked, lat_mesh_model_cop_masked,
        u_interp_cop_masked, v_interp_cop_masked,
        lon_bat, lat_bat, h,
        pdt, SCALE, WIDTH,
        title_name, fig_name,
        1.2, 4.26, 40.5, 43.06,
        mask=[]
    )

