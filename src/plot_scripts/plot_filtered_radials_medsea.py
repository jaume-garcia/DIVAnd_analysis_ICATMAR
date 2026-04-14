import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from datetime import datetime

# -----------------------------------------------------------------------
# Script: plot_filtered_radials_medsea.py
# Purpose: For each hourly snapshot in a 264-step time series, reads the
#          pre-filtered radial HF radar observations (Mediterranean Sea /
#          Catalan Sea domain) and plots the (u, v) radial velocity field
#          colour-coded by radar station. Figures are saved as PNG files.
#
# Pipeline:
#   1. Load the ICATMAR bathymetry grid via Julia.
#   2. Iterate over 264 snapshots.
#   3. For each snapshot, read the filtered radials text file.
#   4. Call the Julia plotting function plot_radials_uv.
# -----------------------------------------------------------------------

# --- Julia bridge setup ---
from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Import the shared Julia utility modules
Main.include("../../utils/reading_obs.jl")
Main.include("../../utils/divand_process.jl")
Main.include("../../utils/visualization_vel.jl")

# --- Python utility modules ---
import sys
sys.path.append('../../utils')

import read_data as rd
import mathematics as ma
import vel_eta as ve
import fig_vel as fig
import HFradar_data as hf

# --- Radar station dictionary (name → antenna flag) ---
radar_dict = hf.radars_cat()

# --- Plot parameters ---
pdt    = 1        # Arrow decimation factor
SCALE  = 15       # Quiver arrow scale
SCALE1 = 15       # Alternative quiver scale (reserved for observations)
WIDTH  = 0.0025   # Quiver arrow width
x_min  = 1.0      # Longitude axis minimum (°)
x_max  = 4.5      # Longitude axis maximum (°)
y_min  = 40.      # Latitude axis minimum (°)
y_max  = 43.5     # Latitude axis maximum (°)

# ------------------------------------------------------------------------------------
# --- Load the ICATMAR bathymetry and domain grid ---
file_icatmar = "../../data/bathy.nc"
mask, h, pm, pn, xi, yi = Main.grid_icatmar(file_icatmar)

# ------------------------------------------------------------------------------------
# --- Main loop: iterate over all hourly snapshots ---
for i in range(264):

    print("ITERATION ... ", i)

    # Path to the filtered radial observations file for snapshot i
    file_txt = (
        "/../../data/january_2026/radials_10_days/filtered_radials/"
        "filtered_radials6_" + str(i).zfill(3) + ".txt"
    )

    # Read the filtered radials into a DataFrame (space-separated, no header)
    df = pd.read_csv(
        file_txt,
        names=['obs_lon', 'obs_lat', 'u_radar', 'v_radar',
               'angle_bearing', 'vel_radar', 'angle_direction', 'flag_radar'],
        skiprows=0,
        sep=' ',
        skipinitialspace=True
    )

    # Convert pandas Series to NumPy arrays before passing to Julia
    obs_lon         = df['obs_lon'].values
    obs_lat         = df['obs_lat'].values
    u_radar         = df['u_radar'].values
    v_radar         = df['v_radar'].values
    angle_bearing   = df['angle_bearing'].values
    vel_radar       = df['vel_radar'].values
    angle_direction = df['angle_direction'].values
    flag_radar      = df['flag_radar'].values

    # Figure title and output path for this snapshot
    title_name2  = ("Synthetic radar observations filtered (Copernicus) - "
                    + str(i).zfill(3) + " hours since 05/02/2025")
    outfile_name2 = (
        "../../figures/january_2026/medsea_radials/"
        "Copernicus_rad_obs_" + str(i).zfill(3) + "_h.png"
    )

    # Call the Julia plotting function: quiver plot of (u, v) radial components
    # colour-coded by antenna flag
    Main.plot_radials_uv(
        xi, yi, mask,
        obs_lon, obs_lat, u_radar, v_radar,
        flag_radar, radar_dict,
        SCALE, WIDTH,
        title_name2,
        x_min, x_max, y_min, y_max,
        outfile_name2
    )
