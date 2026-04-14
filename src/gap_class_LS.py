"""
Script for analyzing gaps in HF radar total velocity fields.
Classifies data gaps by size and visualizes coverage patterns over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colorbar
import matplotlib.colors as mcolors
import sys
import os
from scipy import ndimage

from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("../utils/reading_obs.jl")
Main.include("../utils/divand_process.jl")

sys.path.append('../utils')
import HFradar_data as hf
import read_data as rd
import mathematics as ma

# Initialize HF radar processor
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# BATHYMETRY - Load ICATMAR grid and mask
file_icatmar = "../data/bathy.nc"

# Read grid data: mask (land/water), h (elevation), pm/pn (resolution matrices), xi/yi (coordinate grids)
mask, h, pm, pn, xi, yi = Main.grid_icatmar(file_icatmar)

# ------------------------------------------------------------------------------------
# LS GRID - Load HF radar total velocity grid

grid_file = "../data/hfradar_totals_grid_icatmar.nc"
# Read grid from NetCDF: returns 1D coordinate arrays and 2D meshgrids
lon_LS, lat_LS, lon_grid, lat_grid, mask_antena = processor.read_grid_from_netcdf(grid_file)

# Create 2D meshgrid for LS grid (note: indexing matches processor output)
lat_grid_LS, lon_grid_LS = np.meshgrid(lat_LS, lon_LS)

# BEGU, CREU, TOSS radar coverage (antennas indices 1-4)
mask_group1 = np.sum(mask_antena[1:4, :, :], axis=0)

# AREN, GNST, PBCN radar coverage (antennas indices 5-7)
mask_group2 = np.sum(mask_antena[5:7, :, :], axis=0)

# Combine both groups to get total radar coverage mask
mask_total = mask_group1 + mask_group2
mask_total[mask_total >= 1] = 1  # Binary mask: 1 where any radar has coverage

# Transpose and convert to float for compatibility
mask_total_t = mask_total.T
mask_total_t_float = mask_total_t.astype(np.float64)
mask_total_t_float[mask_total_t_float == 0.0] = np.nan  # No coverage → NaN
mask_total_t = mask_total_t_float

print("SIZE MASK TOTAL =", mask_total_t.shape)

# Interpolate bathymetry mask to LS grid for land/water identification
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

# ------------------------------------------------------------------------------------

def plot_colormesh(xi, yi, diff, mask, title_name, outfile_name):
    """
    Creates a colormesh plot of a 2D field using Cartopy projection.
    
    Parameters:
    -----------
    xi, yi : ndarray
        2D longitude and latitude grids
    diff : ndarray
        2D data field to plot (e.g., point count, differences)
    mask : ndarray
        Land/water mask (True/False for water)
    title_name : str
        Plot title
    outfile_name : str
        Output file path for the figure
    """
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    
    # Plot the main data field
    pc = ax.pcolormesh(xi, yi, diff, shading="auto", cmap="viridis")
    cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cb.set_label("Nº points")
    
    # Overlay land mask
    water_mask = mask > 0.5
    land_mask = mask <= 0.5
    ax.contourf(xi, yi, land_mask, levels=[0.5, 1.0], cmap="copper", alpha=1.0)
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=0.5, alpha=0.5, linestyle="--")
    
    ax.set_title(title_name, fontsize=14, pad=20)
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_extent([1.2, 4.26, 40.5, 43.06])  # Catalan Sea extent
    
    plt.savefig(outfile_name)
    plt.close()

# ------------------------------------------------------------------------------------

def putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS, 
                               u_array, v_array):
    """
    Assigns data points to the LS (Least Squares) grid.
    
    For each data point, finds the nearest grid cell and assigns the velocity values.
    
    Parameters:
    -----------
    lon_array, lat_array : ndarray
        Coordinates of data points
    lon_grid_LS, lat_grid_LS : ndarray
        2D grid coordinates
    u_array, v_array : ndarray
        Velocity components at data points
    
    Returns:
    --------
    tuple
        u_LS_2d, v_LS_2d : 2D arrays with velocities assigned to grid points (NaN where no data)
    """
    tolerance = 1e-6
    
    u_LS_2d = np.full(lon_grid_LS.shape, np.nan)
    v_LS_2d = np.full(lon_grid_LS.shape, np.nan)
    
    points_found = 0
    points_not_found = 0
    
    for j in range(len(lon_array)):
        # Find closest grid index using Euclidean distance
        distances = np.sqrt((lon_grid_LS - lon_array[j])**2 + 
                           (lat_grid_LS - lat_array[j])**2)
        min_distance = np.min(distances)
        idx_closest = np.unravel_index(np.argmin(distances), distances.shape)
        idx_lon, idx_lat = idx_closest
        
        # Verify if the point is exactly on the grid (within tolerance)
        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance and 
                        abs(lat_closest - lat_array[j]) < tolerance)
        
        if is_grid_point:
            points_found += 1
        else:
            points_not_found += 1
            
            # Print warnings for first few points not on grid
            if points_not_found <= 5:
                print(f"WARNING: Point {j} not exactly on grid")
                print(f"  Distance: {min_distance}")
                print(f"  Target: ({lon_array[j]}, {lat_array[j]})")
                print(f"  Closest: ({lon_closest}, {lat_closest})")
        
        # Assign velocity values to the grid cell
        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]
    
    return u_LS_2d, v_LS_2d

# ------------------------------------------------------------------------------------

def classify_gaps(speed, water_mask, n_clusters=20, small_threshold=5, large_threshold=10):
    """
    Creates gap clusters in a velocity field and classifies them by size.
    
    Adds synthetic gaps randomly, then labels connected components and
    categorizes them as small or large based on pixel count.
    
    Parameters:
    -----------
    speed : np.ndarray
        2D velocity magnitude field (modified in-place by adding NaN gaps)
    water_mask : np.ndarray
        Boolean mask for water cells (True = water)
    n_clusters : int
        Number of gap clusters to create
    small_threshold : int
        Gaps with fewer pixels than this → mask_small
    large_threshold : int
        Gaps with more pixels than this → mask_large
    
    Returns:
    --------
    gap_mask : np.ndarray (bool)
        Mask of all gaps (NaN locations in water)
    mask_small : np.ndarray (bool)
        Mask of small gaps (size < small_threshold)
    mask_large : np.ndarray (bool)
        Mask of large gaps (size > large_threshold)
    labels : np.ndarray (int)
        Labels for each connected region
    sizes : np.ndarray
        Size (in pixels) of each labeled region
    """
    nx, ny = speed.shape

    # Create synthetic gap clusters at random locations
    for _ in range(n_clusters):
        cx = np.random.randint(10, nx - 10)   # Center x
        cy = np.random.randint(10, ny - 10)   # Center y
        r = np.random.randint(4, 10)          # Radius
        sub = speed[cx-r:cx+r, cy-r:cy+r]
        rand_mask = np.random.rand(*sub.shape) < 0.4  # 40% probability of gap
        sub[rand_mask] = np.nan

    # Detect and label connected regions of gaps
    gap_mask = np.isnan(speed) & (water_mask > 0.5)
    structure = np.ones((3, 3))  # 8-connectivity (including diagonals)
    labels, _ = ndimage.label(gap_mask, structure=structure)

    # Calculate size (pixel count) of each region
    nlabels = labels.max()
    sizes = np.array(ndimage.sum(gap_mask, labels, range(1, nlabels + 1)))

    # Classify regions by size
    small_ids = np.where(sizes < small_threshold)[0] + 1
    large_ids = np.where(sizes > large_threshold)[0] + 1
    mask_small = np.isin(labels, small_ids)
    mask_large = np.isin(labels, large_ids)

    return gap_mask, mask_small, mask_large, labels, sizes

# ------------------------------------------------------------------------------------

def plot_gaps(lon, lat, speed, mask, domain_mask, mask_small, mask_large, outfile_name):
    """
    Visualizes velocity field with gaps and classifies gaps by size.
    
    Creates a two-panel figure:
    - Left: velocity magnitude field with gaps
    - Right: gap classification (small vs large)
    
    Parameters:
    -----------
    lon, lat : ndarray
        2D coordinate grids
    speed : ndarray
        2D velocity magnitude field (may contain NaNs)
    mask : ndarray
        Land/water mask (>0.5 = water)
    domain_mask : ndarray
        Radar coverage domain mask (NaN where no coverage)
    mask_small : ndarray (bool)
        Small gaps mask
    mask_large : ndarray (bool)
        Large gaps mask
    outfile_name : str
        Output file path for the figure
    """
    nx, ny = speed.shape

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    # ---- Subplot 0: velocity field with gaps ----
    pm0 = ax0.pcolormesh(lon, lat, speed,
                         cmap="viridis",
                         transform=ccrs.PlateCarree())
    plt.colorbar(pm0, ax=ax0, label="Speed (m/s)", shrink=0.7)

    land_mask = (mask <= 0.5).astype(float)
    ax0.contourf(lon, lat, land_mask, levels=[0.5, 1.0],
                 cmap="copper", alpha=1.0,
                 transform=ccrs.PlateCarree())

    ax0.set_title("Field with gaps")
    ax0.set_xlabel("Longitude (°)", fontsize=12)
    ax0.set_ylabel("Latitude (°)", fontsize=12)
    ax0.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
    ax0.gridlines(draw_labels=True, dms=True,
                  x_inline=False, y_inline=False,
                  linewidth=0.5, alpha=0.5, linestyle="--")

    # ---- Subplot 1: classified gaps ----
    # Build numeric array: 0=no gap, 1=small, 2=large
    land_mask = (mask <= 0.5).astype(float)
    water_mask = mask > 0.5
    
    # domain_mask indicates LS coverage: 1 where available, NaN outside
    ls_domain = np.isfinite(domain_mask) & water_mask  # Water cells WITHIN LS domain

    gap_class = np.full((nx, ny), np.nan)   # Outside LS domain → NaN
    gap_class[ls_domain] = 0               # LS water without gap → gray
    gap_class[mask_small & ls_domain] = 1  # Small gap → blue
    gap_class[mask_large & ls_domain] = 2  # Large gap → red

    cmap_gaps = mcolors.ListedColormap(["lightgrey", "blue", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_gaps.N)

    pm1 = ax1.pcolormesh(lon, lat, gap_class, cmap=cmap_gaps, norm=norm, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(pm1, ax=ax1, shrink=0.7)
    
    ax1.contourf(lon, lat, land_mask, levels=[0.5, 1.0],
                 cmap="copper", alpha=1.0, transform=ccrs.PlateCarree())
    
    cbar1.set_ticks([0, 1, 2])
    cbar1.set_ticklabels(["No gap", "Small (<5)", "Large (>10)"])

    ax1.set_title("Classified gaps")
    ax1.set_xlabel("Longitude (°)", fontsize=12)
    ax1.set_ylabel("Latitude (°)", fontsize=12)
    ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  linewidth=0.5, alpha=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(outfile_name, dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------------

# Main processing loop
n_times = 264  # Number of time steps (11 days × 24 hours)

# Initialize array to count valid points per grid cell over time
point_count_grid = np.zeros((n_times, *lon_grid_LS.shape), dtype=int)

for i in range(n_times):
    print(f"ITERATION ... {i+1}")
    
    # TOTAL VELOCITIES (L3) - Read Least Squares total velocity file
    # file_tuv = f"../data/february_2025/totals_10_days/medsea_totals_{i:03d}_all_grid.txt"  # Alternative
    file_tuv = f"../data/january_2026/totals_10_days/medsea_totals_{i:03d}_all_grid.txt"
    
    # Read tab-separated data (lines starting with '#' are comments)
    df = pd.read_csv(file_tuv, sep='\t', comment='#')
    
    # Extract data columns
    lon_array = df['longitude'].values
    lat_array = df['latitude'].values
    u_array = df['u_total'].values
    v_array = df['v_total'].values
    mod_vel_array = df['modulo'].values
    angle_array = df['angulo'].values
    gdop_array = df['gdop'].values
    
    # Filter by GDOP (Geometric Dilution of Precision) and velocity magnitude
    # GDOP ≤ 2.0 indicates good geometric configuration
    # Velocity ≤ 1.2 m/s removes unrealistic spikes
    valid_index = (gdop_array <= 2.0) & (mod_vel_array <= 1.2)
    
    # Apply filter
    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array = angle_array[valid_index]
    
    # Assign velocity values to LS grid cells
    u_LS_2d, v_LS_2d = putting_points_to_LS_grid(lon_array, lat_array,
                                                   lon_grid_LS, lat_grid_LS,
                                                   u_array, v_array)
    
    # Mark grid cells with valid data (1) vs NaN (0)
    point_count_grid[i, :, :] = ~(np.isnan(u_LS_2d) | np.isnan(v_LS_2d))
    
    # Compute velocity magnitude for gap analysis
    mod_vel_2d = np.sqrt(u_LS_2d**2 + v_LS_2d**2)
    
    print(f"Valid points in this iteration: {np.sum(point_count_grid[i, :, :])}")
    
    # Classify gaps in the velocity field
    gap_mask, mask_small, mask_large, labels, sizes = classify_gaps(
        mod_vel_2d, mask_interp,
        n_clusters=20, small_threshold=5, large_threshold=10
    )
    
    # Generate and save gap visualization
    outfile_name = f"../figures/january_2026/gaps/time_series_vel_mag/gaps_vel_{i:03d}"
    
    plot_gaps(lon_grid_LS, lat_grid_LS, mod_vel_2d, mask_interp,
              mask_total_t, mask_small, mask_large, outfile_name)

# ------------------------------------------------------------------------------------
# POST-PROCESSING - Compute temporal statistics

# Sum point counts across all time steps
total_point_count = np.sum(point_count_grid, axis=0)

# Convert to float and set zero counts to NaN for proper visualization
total_point_count_float = total_point_count.astype(float)
total_point_count_float[total_point_count == 0] = np.nan

# Apply radar coverage mask (NaN outside coverage area)
total_point_count_float = total_point_count_float * mask_total_t

print("\n=== FINAL SUMMARY ===")
print(f"Maximum points at any grid cell: {np.max(total_point_count)}")
print(f"Number of cells with at least one valid point: {np.sum(total_point_count > 0)}")
print(f"Average points per cell (where data exists): {np.mean(total_point_count[total_point_count > 0])}")

# Generate final map showing number of valid points used across the time series
title_name6 = "Number of total valid points used in the analysis through time series"
outfile_name6 = "../figures/january_2026/map_valid_points.png"

plot_colormesh(lon_grid_LS, lat_grid_LS, total_point_count_float, mask_interp, title_name6, outfile_name6)
