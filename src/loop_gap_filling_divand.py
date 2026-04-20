"""
Gap filling validation for DIVAnd reconstruction.
Compares DIVAnd (radial and total) against Copernicus ground truth
specifically in regions where LS data has gaps (NaNs).
"""

import numpy as np
import pandas as pd
import sys
import os

from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("../utils/reading_obs.jl")
Main.include("../utils/divand_process.jl")

sys.path.append("../utils")
import HFradar_data as hf
import vel_eta as ve
import read_data as rd
import mathematics as ma

processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------
bathy_file = "../data/bathy.nc"
mask, h, pm, pn, xi, yi = Main.grid_icatmar(bathy_file)

# ------------------------------------------------------------------------------------
# LS GRID
# ------------------------------------------------------------------------------------

grid_file = "../data/hfradar_totals_grid_icatmar.nc"
lon_LS, lat_LS, lon_grid, lat_grid, antenna_mask = processor.read_grid_from_netcdf(grid_file)
lat_grid_LS, lon_grid_LS = np.meshgrid(lat_LS, lon_LS)

# Group 1: BEGU, CREU, TOSS coverage
mask_group1 = np.sum(antenna_mask[1:4, :, :], axis=0)

# Group 2: AREN, GNST, PBCN coverage
mask_group2 = np.sum(antenna_mask[5:7, :, :], axis=0)

# Total coverage mask (union of all radars)
total_mask = mask_group1 + mask_group2
total_mask[total_mask >= 1] = 1

total_mask_t = total_mask.T
total_mask_t_float = total_mask_t.astype(np.float64)
total_mask_t_float[total_mask_t_float == 0.0] = np.nan
total_mask_t = total_mask_t_float

print("SIZE TOTAL MASK =", total_mask_t.shape)

# Interpolate bathymetry mask to LS grid
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

# ------------------------------------------------------------------------------------
# COPERNICUS GROUND TRUTH
# ------------------------------------------------------------------------------------

copernicus_file = "../data/january_2026/data_medsea/all_data_january_2026.nc"

lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(
    copernicus_file, "lon", "lat", "u_data", "v_data", "0:"
)

lat_mesh_cop, lon_mesh_cop = np.meshgrid(lat_cop, lon_cop)
lon_mesh_cop = lon_mesh_cop * (180 / np.pi)
lat_mesh_cop = lat_mesh_cop * (180 / np.pi)

FILL_VALUE = 1.0e20
u_cop_model = u_cop.astype(np.float64)
v_cop_model = v_cop.astype(np.float64)
u_cop_model[np.abs(u_cop_model) >= FILL_VALUE] = np.nan
v_cop_model[np.abs(v_cop_model) >= FILL_VALUE] = np.nan

print("COPERNICUS data loaded")

# ------------------------------------------------------------------------------------
# DIVAND RECONSTRUCTION DATA (RADIAL AND TOTAL)
# ------------------------------------------------------------------------------------

divand_file = "../data/january_2026/divand/divand_field.nc"
lon_divand, lat_divand, time_divand, u_radial, v_radial, u_total, v_total = Main.read_divand_file(divand_file)

# ------------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------------

n_times = 264
stats_time_series = np.zeros((n_times, 25))

for i in range(n_times):

    print(f"ITERATION ... {i}")

    # Copernicus timestep
    u_cop_ts = u_cop_model[i, :, :]
    v_cop_ts = v_cop_model[i, :, :]

    # DIVAnd timestep (radial and total)
    u_radial_ts = u_radial[i, :, :]
    v_radial_ts = v_radial[i, :, :]
    u_total_ts = u_total[i, :, :]
    v_total_ts = v_total[i, :, :]

    # LS total velocities (Level 3)
    totals_file = f"../data/january_2026/totals_10_days/medsea_totals_{str(i).zfill(3)}_all_grid.txt"

    df = pd.read_csv(totals_file, sep='\t', comment='#', skipinitialspace=True)

    lon_array = df["longitude"].values
    lat_array = df["latitude"].values
    u_array = df["u_total"].values
    v_array = df["v_total"].values
    speed_array = df["speed"].values
    angle_array = df["direction"].values
    gdop_array = df["gdop"].values

    # Filter by GDOP and maximum speed
    valid_idx = (gdop_array <= 2.0) & (speed_array <= 1.2)

    lon_array = lon_array[valid_idx]
    lat_array = lat_array[valid_idx]
    u_array = u_array[valid_idx]
    v_array = v_array[valid_idx]
    speed_array = speed_array[valid_idx]
    angle_array = angle_array[valid_idx]

    print("LS totals processed")
    
    # Grid LS velocities
    u_LS_2d, v_LS_2d = ve.putting_points_to_LS_grid(
        lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array
    )
    
    # Interpolate Copernicus to LS grid
    u_cop_interp, v_cop_interp = ma.interp_grid_vel(
        lon_mesh_cop, lat_mesh_cop, u_cop_ts, v_cop_ts,
        lon_grid_LS, lat_grid_LS, mask_coarse_grid=[], points_mode=False
    )
    
    # Speed magnitudes
    speed_cop = np.sqrt(u_cop_interp**2 + v_cop_interp**2)
    speed_rad = np.sqrt(u_radial_ts**2 + v_radial_ts**2)
    speed_total = np.sqrt(u_total_ts**2 + v_total_ts**2)
    speed_LS = np.sqrt(u_LS_2d**2 + v_LS_2d**2)

    # Apply radar coverage mask
    u_cop_masked = u_cop_interp * total_mask_t
    v_cop_masked = v_cop_interp * total_mask_t
    speed_cop_masked = speed_cop * total_mask_t

    u_radial_masked = u_radial_ts * total_mask_t
    v_radial_masked = v_radial_ts * total_mask_t
    speed_rad_masked = speed_rad * total_mask_t

    u_total_masked = u_total_ts * total_mask_t
    v_total_masked = v_total_ts * total_mask_t
    speed_total_masked = speed_total * total_mask_t

    speed_LS_masked = speed_LS * total_mask_t

    # Identify gap pixels: NaNs in LS but valid in DIVAnd and Copernicus
    gaps_mask = (np.isnan(speed_LS_masked) & 
                 ~(np.isnan(speed_cop_masked) | np.isnan(speed_rad_masked) | np.isnan(speed_total_masked)))
    
    n_gaps = np.sum(gaps_mask)

    # Extract gap pixel values
    u_cop_gaps = u_cop_masked[gaps_mask]
    v_cop_gaps = v_cop_masked[gaps_mask]
    speed_cop_gaps = speed_cop_masked[gaps_mask]

    u_radial_gaps = u_radial_masked[gaps_mask]
    v_radial_gaps = v_radial_masked[gaps_mask]
    speed_rad_gaps = speed_rad_masked[gaps_mask]

    u_total_gaps = u_total_masked[gaps_mask]
    v_total_gaps = v_total_masked[gaps_mask]
    speed_total_gaps = speed_total_masked[gaps_mask]

    # Compute differences (reconstruction - ground truth)
    diff_u_rad = u_radial_gaps - u_cop_gaps
    diff_v_rad = v_radial_gaps - v_cop_gaps
    diff_speed_rad = speed_rad_gaps - speed_cop_gaps

    diff_u_total = u_total_gaps - u_cop_gaps
    diff_v_total = v_total_gaps - v_cop_gaps
    diff_speed_total = speed_total_gaps - speed_cop_gaps
    
    # Compute bias (mean difference)
    bias_u_rad = np.nanmean(diff_u_rad)
    bias_v_rad = np.nanmean(diff_v_rad)
    bias_speed_rad = np.nanmean(diff_speed_rad)

    bias_u_total = np.nanmean(diff_u_total)
    bias_v_total = np.nanmean(diff_v_total)
    bias_speed_total = np.nanmean(diff_speed_total)
    
    # Compute RMSE (Root Mean Square Error)
    rmse_u_rad = np.sqrt(np.nanmean((u_radial_gaps - u_cop_gaps)**2))
    rmse_v_rad = np.sqrt(np.nanmean((v_radial_gaps - v_cop_gaps)**2))
    total_rmse_rad = np.sqrt(np.nansum((u_radial_gaps - u_cop_gaps)**2 + 
                                       (v_radial_gaps - v_cop_gaps)**2) / n_gaps)
    
    rmse_u_total = np.sqrt(np.nanmean((u_total_gaps - u_cop_gaps)**2))
    rmse_v_total = np.sqrt(np.nanmean((v_total_gaps - v_cop_gaps)**2))
    total_rmse_total = np.sqrt(np.nansum((u_total_gaps - u_cop_gaps)**2 + 
                                         (v_total_gaps - v_cop_gaps)**2) / n_gaps)
    
    # Compute MAE (Mean Absolute Error)
    mae_u_rad = np.nanmean(np.abs(u_radial_gaps - u_cop_gaps))
    mae_v_rad = np.nanmean(np.abs(v_radial_gaps - v_cop_gaps))
    mae_u_total = np.nanmean(np.abs(u_total_gaps - u_cop_gaps))
    mae_v_total = np.nanmean(np.abs(v_total_gaps - v_cop_gaps))

    total_mae_rad = np.nanmean(np.abs(speed_rad_gaps - speed_cop_gaps))
    total_mae_total = np.nanmean(np.abs(speed_total_gaps - speed_cop_gaps))
    
    # Compute STD of differences
    std_u_rad = np.nanstd(diff_u_rad)
    std_v_rad = np.nanstd(diff_v_rad)
    std_speed_rad = np.nanstd(diff_speed_rad)

    std_u_total = np.nanstd(diff_u_total)
    std_v_total = np.nanstd(diff_v_total)
    std_speed_total = np.nanstd(diff_speed_total)
    
    # Print results
    print(f"\n=== GAP FILLING RESULTS ===")
    print(f"  Number of gap pixels: {n_gaps}")
    print(f"  bias_u_rad   = {bias_u_rad:.6f} m/s  |  bias_u_total   = {bias_u_total:.6f} m/s")
    print(f"  bias_v_rad   = {bias_v_rad:.6f} m/s  |  bias_v_total   = {bias_v_total:.6f} m/s")
    print(f"  bias_speed_rad = {bias_speed_rad:.6f} m/s  |  bias_speed_total = {bias_speed_total:.6f} m/s")
    print("=========================================================\n")
    print(f"  RMSE_u_rad  = {rmse_u_rad:.6f} m/s  |  RMSE_u_total  = {rmse_u_total:.6f} m/s")
    print(f"  RMSE_v_rad  = {rmse_v_rad:.6f} m/s  |  RMSE_v_total  = {rmse_v_total:.6f} m/s")
    print(f"  RMSE_total_rad = {total_rmse_rad:.6f} m/s  |  RMSE_total_total = {total_rmse_total:.6f} m/s")
    print(f"\n  MAE_u_rad   = {mae_u_rad:.6f} m/s  |  MAE_u_total   = {mae_u_total:.6f} m/s")
    print(f"  MAE_v_rad   = {mae_v_rad:.6f} m/s  |  MAE_v_total   = {mae_v_total:.6f} m/s")
    print(f"  MAE_total_rad = {total_mae_rad:.6f} m/s  |  MAE_total_total = {total_mae_total:.6f} m/s")
    print("=========================================================\n")

    # Store statistics in time series matrix
    stats_time_series[i, :] = [
        i, 
        bias_u_rad, bias_v_rad, bias_speed_rad, 
        bias_u_total, bias_v_total, bias_speed_total,
        rmse_u_rad, rmse_v_rad, total_rmse_rad,
        rmse_u_total, rmse_v_total, total_rmse_total,
        mae_u_rad, mae_v_rad, total_mae_rad,
        mae_u_total, mae_v_total, total_mae_total,
        std_u_rad, std_v_rad, std_speed_rad,
        std_u_total, std_v_total, std_speed_total
    ]
    
# Save time series to file
output_file = "../data/january_2026/gaps_data/gaps_avg_time_series.txt"
np.savetxt(output_file, stats_time_series, fmt="%.6f", delimiter=" ")

print(f"ASCII file generated: {output_file}")
    


