"""
Gap filling validation with size-based classification.
Classifies gaps into small (<4 pixels) and large (≥4 pixels) categories,
then evaluates DIVAnd reconstruction performance separately for each category.
"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_fill_holes, label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from datetime import datetime, timedelta

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

# Total coverage mask
total_mask = mask_group1 + mask_group2
total_mask[total_mask >= 1] = 1

total_mask_t = total_mask.T
total_mask_t_float = total_mask_t.astype(np.float64)
total_mask_t_float[total_mask_t_float == 0.0] = np.nan
total_mask_t = total_mask_t_float

print("SIZE TOTAL MASK =", total_mask_t.shape)

# Interpolate bathymetry mask to LS grid
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, coarse_grid_mask=[])

# ------------------------------------------------------------------------------------
# GAP CLASSIFICATION PARAMETERS
# ------------------------------------------------------------------------------------

SMALL_THRESHOLD = 4   # gaps with fewer pixels than this → small
                      # gaps with >= SMALL_THRESHOLD pixels → large
CLOSING_SIZE = 4      # size of square structuring element for binary_closing
                      # (e.g. 3 → 3×3, 5 → 5×5, 10 → 10×10)


def classify_gaps(speed_2d, water_mask, small_threshold=SMALL_THRESHOLD, closing_size=CLOSING_SIZE):
    """
    Classify gaps by size using morphological closing and connected components.
    
    Parameters
    ----------
    speed_2d        : 2D array — speed magnitude on LS grid (NaN where no data)
    water_mask      : 2D array — water domain mask (> 0.5 = water)
    small_threshold : int — pixel-count boundary between small and large gaps
    closing_size    : int — side length of square structuring element

    Returns
    -------
    gap_mask   : bool array — all detected gap pixels
    small_mask : bool array — small gaps (size < small_threshold)
    large_mask : bool array — large gaps (size >= small_threshold)
    labels     : int array  — connected-component labels
    sizes      : 1D array   — pixel count of each labelled region
    """
    # Ensure inputs are strictly 2D
    speed_2d = np.squeeze(speed_2d)
    water_mask = np.squeeze(water_mask)

    # Binary mask: True where speed data is available
    binary_mask = ~np.isnan(speed_2d)

    # Morphological closing to detect enclosed regions
    structure = np.ones((closing_size, closing_size))
    closed = binary_closing(binary_mask, structure=structure)
    filled = binary_fill_holes(closed)

    # Gaps are enclosed pixels with no data, restricted to water domain
    gap_mask = filled & (~binary_mask) & (water_mask > 0.5)

    # Label connected components
    label_structure = np.ones((3, 3))
    labels, _ = label(gap_mask, structure=label_structure)

    nlabels = labels.max()
    if nlabels == 0:
        sizes = np.array([])
        return gap_mask, np.zeros_like(gap_mask, dtype=bool), np.zeros_like(gap_mask, dtype=bool), labels, sizes

    sizes = np.array(ndimage.sum(gap_mask, labels, range(1, nlabels + 1)))

    small_ids = np.where(sizes < small_threshold)[0] + 1
    large_ids = np.where(sizes >= small_threshold)[0] + 1
    small_mask = np.isin(labels, small_ids)
    large_mask = np.isin(labels, large_ids)

    return gap_mask, small_mask, large_mask, labels, sizes


def plot_gaps(lon, lat, speed, mask, domain_mask, small_mask, large_mask, output_file, date):
    """
    Visualize velocity field with size-classified gaps.
    
    Parameters
    ----------
    speed        : 2D array — velocity field (can contain NaNs)
    mask         : 2D array — water/land mask (>0.5 = water)
    domain_mask  : 2D array — LS domain mask (1 inside, NaN outside)
    small_mask   : bool array — small gaps
    large_mask   : bool array — large gaps
    output_file  : str — output file path
    lon, lat     : 2D arrays — grid coordinates
    date         : datetime — timestamp
    """
    radars = [
        {"name": "CREU",  "lat": 42.31858749715234, "lon": 3.315556921342478},
        {"name": "BEGU",  "lat": 41.96677154480271, "lon": 3.231267423150908},
        {"name": "TOSS",  "lat": 41.71563535258847, "lon": 2.93470263732462},
        {"name": "AREN",  "lat": 41.57831210624108, "lon": 2.562880818691122},
        {"name": "PBCN",  "lat": 41.33436385026707, "lon": 2.17109631526204},
        {"name": "PGNST", "lat": 41.25567563411607, "lon": 1.922701886580638},
    ]

    offsets = {
        "CREU":  (0.46,  0.04),
        "BEGU":  (0.46,  0.04),
        "TOSS":  (0.46,  0.07),
        "AREN":  (0.47,  0.05),
        "PBCN":  (0.41,  0.12),  
        "PGNST": (0.5,   0.08),
    }

    def plot_radars(ax):
        for r in radars:
            dx, dy = offsets.get(r["name"], (0.02, 0.02))
            ax.plot(r["lon"], r["lat"],
                    marker=".", markersize=6, markeredgewidth=0.8, color="magenta", markeredgecolor="black",
                    transform=ccrs.PlateCarree(), zorder=5)
            ax.text(r["lon"] - dx, r["lat"] + dy, r["name"],
                    fontsize=7, fontweight="bold", color="black",
                    transform=ccrs.PlateCarree(), zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="black", linewidth=0.8))

    nx, ny = speed.shape
    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Mercator())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Mercator())

    fig.suptitle(date.strftime("%Y-%m-%d  %H:%M UTC"), fontsize=14, fontweight="bold", y=1.01)

    # Subplot 0: velocity field with gaps
    pm0 = ax0.pcolormesh(lon, lat, speed, cmap="viridis", transform=ccrs.PlateCarree())
    plt.colorbar(pm0, ax=ax0, label="Speed (m/s)", shrink=0.7)
    land_mask = (mask <= 0.5).astype(float)
    ax0.contourf(lon, lat, land_mask, levels=[0.5, 1.0], cmap="copper", alpha=1.0, transform=ccrs.PlateCarree())
    plot_radars(ax0)
    ax0.set_title("LS Velocities (magnitude)")
    ax0.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    gl0 = ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5, linestyle="--")
    gl0.right_labels = False

    # Subplot 1: classified gaps
    land_mask = (mask <= 0.5).astype(float)
    water_mask = mask > 0.5
    has_data = ~np.isnan(speed) & water_mask

    gap_class = np.full((nx, ny), np.nan)
    gap_class[water_mask] = 0
    gap_class[has_data] = 1
    gap_class[small_mask & water_mask] = 2
    gap_class[large_mask & water_mask] = 3

    cmap_gaps = mcolors.ListedColormap(["white", "yellow", "blue", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_gaps.N)
    pm1 = ax1.pcolormesh(lon, lat, gap_class, cmap=cmap_gaps, norm=norm, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(pm1, ax=ax1, shrink=0.7)
    ax1.contourf(lon, lat, land_mask, levels=[0.5, 1.0], cmap="copper", alpha=1.0, transform=ccrs.PlateCarree())
    plot_radars(ax1)
    cbar1.set_ticks([0, 1, 2, 3])
    cbar1.set_ticklabels(["No data", "HF-Radar data", "Small gap (<4)", "Large gap (>=4)"])
    ax1.set_title("Gap classification")
    ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5, linestyle="--")
    gl1.right_labels = False

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def compute_stats(u_ref, v_ref, speed_ref, u_fld, v_fld, speed_fld, gap_mask):
    """
    Compute bias, RMSE, MAE and STD for u, v and speed magnitude
    at the pixels selected by gap_mask.
    
    Parameters
    ----------
    u_ref / v_ref / speed_ref : reference field (Copernicus, masked to LS domain)
    u_fld / v_fld / speed_fld : field to evaluate (DIVAnd-radial or DIVAnd-total)
    gap_mask                  : boolean mask selecting gap pixels of a given size class

    Returns
    -------
    dict with statistics
    """
    N = int(np.sum(gap_mask))
    if N == 0:
        nan12 = [np.nan] * 12
        return dict(zip(
            ["N", "avg_u", "avg_v", "avg_speed",
             "rmse_u", "rmse_v", "rmse_total",
             "mae_u",  "mae_v",  "mae_total",
             "std_u",  "std_v",  "std_total"], [0] + nan12))

    u_r = u_ref[gap_mask]
    v_r = v_ref[gap_mask]
    sp_r = speed_ref[gap_mask]

    u_f = u_fld[gap_mask]
    v_f = v_fld[gap_mask]
    sp_f = speed_fld[gap_mask]

    avg_u = np.nanmean(u_f - u_r)
    avg_v = np.nanmean(v_f - v_r)
    avg_speed = np.nanmean(sp_f - sp_r)

    rmse_u = np.sqrt(np.nanmean((u_f - u_r)**2))
    rmse_v = np.sqrt(np.nanmean((v_f - v_r)**2))
    rmse_total = np.sqrt(np.nansum((u_f - u_r)**2 + (v_f - v_r)**2) / N)

    mae_u = np.nanmean(np.abs(u_f - u_r))
    mae_v = np.nanmean(np.abs(v_f - v_r))
    mae_total = np.nanmean(np.abs(sp_f - sp_r))

    std_u = np.nanstd(u_f - u_r)
    std_v = np.nanstd(v_f - v_r)
    std_total = np.nanstd(sp_f - sp_r)

    return dict(N=N,
                avg_u=avg_u, avg_v=avg_v, avg_speed=avg_speed,
                rmse_u=rmse_u, rmse_v=rmse_v, rmse_total=rmse_total,
                mae_u=mae_u, mae_v=mae_v, mae_total=mae_total,
                std_u=std_u, std_v=std_v, std_total=std_total)


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
# DIVAND RECONSTRUCTION DATA
# ------------------------------------------------------------------------------------

divand_file = "../data/january_2026/divand/divand_field.nc"
lon_divand, lat_divand, time_divand, u_radial, v_radial, u_total, v_total = Main.read_divand_file(divand_file)

# ------------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------------

n_times = 264
start_date = datetime.strptime('2026-01-14 00:00:00', '%Y-%m-%d %H:%M:%S')

# Statistics arrays: [time_index, small_radial×13, small_total×13]
stats_small = np.zeros((n_times, 1 + 13 * 2))
stats_large = np.zeros((n_times, 1 + 13 * 2))

for i in range(n_times):

    print(f"\nITERATION ... {i}")

    # Copernicus timestep
    u_cop_ts = u_cop_model[i, :, :]
    v_cop_ts = v_cop_model[i, :, :]

    # DIVAnd timestep
    u_radial_ts = u_radial[i, :, :]
    v_radial_ts = v_radial[i, :, :]
    u_total_ts = u_total[i, :, :]
    v_total_ts = v_total[i, :, :]

    # LS totals
    totals_file = f"../data/january_2026/totals_10_days/medsea_totals_{str(i).zfill(3)}_all_grid.txt"

    df = pd.read_csv(totals_file, sep='\t', comment='#', skipinitialspace=True)

    lon_array = df["longitude"].values
    lat_array = df["latitude"].values
    u_array = df["u_total"].values
    v_array = df["v_total"].values
    speed_array = df["speed"].values
    gdop_array = df["gdop"].values

    valid_idx = (gdop_array <= 2.0) & (speed_array <= 1.2)
    lon_array = lon_array[valid_idx]
    lat_array = lat_array[valid_idx]
    u_array = u_array[valid_idx]
    v_array = v_array[valid_idx]

    u_LS_2d, v_LS_2d = ve.putting_points_to_LS_grid(
        lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array
    )

    # Interpolate Copernicus to LS grid
    u_cop_interp, v_cop_interp = ma.interp_grid_vel(
        lon_mesh_cop, lat_mesh_cop, u_cop_ts, v_cop_ts,
        lon_grid_LS, lat_grid_LS, coarse_grid_mask=[], points_mode=False
    )

    # Speed magnitudes
    speed_cop = np.sqrt(u_cop_interp**2 + v_cop_interp**2)
    speed_rad = np.sqrt(u_radial_ts**2 + v_radial_ts**2)
    speed_total = np.sqrt(u_total_ts**2 + v_total_ts**2)
    speed_LS = np.sqrt(u_LS_2d**2 + v_LS_2d**2)

    # Apply LS domain mask
    u_cop_m = u_cop_interp * total_mask_t
    v_cop_m = v_cop_interp * total_mask_t
    speed_cop_m = speed_cop * total_mask_t

    u_radial_m = u_radial_ts * total_mask_t
    v_radial_m = v_radial_ts * total_mask_t
    speed_rad_m = speed_rad * total_mask_t

    u_total_m = u_total_ts * total_mask_t
    v_total_m = v_total_ts * total_mask_t
    speed_total_m = speed_total * total_mask_t

    speed_LS_m = speed_LS * total_mask_t

    # Classify gaps by size
    _, small_mask, large_mask, _, sizes = classify_gaps(speed_LS_m, mask_interp)

    print(f"  Gap pixels — small: {small_mask.sum()}  |  large: {large_mask.sum()}")

    # Plot gap classification (optional, uncomment to generate plots)
    current_date = start_date + timedelta(hours=int(i))
    # output_plot = f"../figures/january_2026/gaps/time_series_vel_mag/gaps_vel_{str(i).zfill(3)}.png"
    # plot_gaps(lon_grid_LS, lat_grid_LS, speed_LS_m, mask_interp, total_mask_t, small_mask, large_mask, output_plot, current_date)

    # Base valid-data filter: only consider gaps where DIVAnd and Copernicus are available
    base_valid = (~np.isnan(speed_cop_m) & ~np.isnan(speed_rad_m) & ~np.isnan(speed_total_m))

    small_valid = small_mask & base_valid
    large_valid = large_mask & base_valid

    # Statistics for each size class × method
    stats_small_rad = compute_stats(u_cop_m, v_cop_m, speed_cop_m, u_radial_m, v_radial_m, speed_rad_m, small_valid)
    stats_small_tot = compute_stats(u_cop_m, v_cop_m, speed_cop_m, u_total_m, v_total_m, speed_total_m, small_valid)
    stats_large_rad = compute_stats(u_cop_m, v_cop_m, speed_cop_m, u_radial_m, v_radial_m, speed_rad_m, large_valid)
    stats_large_tot = compute_stats(u_cop_m, v_cop_m, speed_cop_m, u_total_m, v_total_m, speed_total_m, large_valid)

    def row(d):
        return [d["N"], d["avg_u"], d["avg_v"], d["avg_speed"],
                d["rmse_u"], d["rmse_v"], d["rmse_total"],
                d["mae_u"], d["mae_v"], d["mae_total"],
                d["std_u"], d["std_v"], d["std_total"]]

    stats_small[i, :] = [i] + row(stats_small_rad) + row(stats_small_tot)
    stats_large[i, :] = [i] + row(stats_large_rad) + row(stats_large_tot)

    # Print summary
    for label, d_rad, d_tot in [("SMALL gaps", stats_small_rad, stats_small_tot),
                                ("LARGE gaps", stats_large_rad, stats_large_tot)]:
        print(f"\n  === {label} ===")
        print(f"  N = {d_rad['N']}  (radial)  |  N = {d_tot['N']}  (total)")
        print(f"  avg_u   rad={d_rad['avg_u']:.6f}  tot={d_tot['avg_u']:.6f}  m/s")
        print(f"  avg_v   rad={d_rad['avg_v']:.6f}  tot={d_tot['avg_v']:.6f}  m/s")
        print(f"  avg_speed rad={d_rad['avg_speed']:.6f}  tot={d_tot['avg_speed']:.6f}  m/s")
        print(f"  RMSE_u  rad={d_rad['rmse_u']:.6f}  tot={d_tot['rmse_u']:.6f}  m/s")
        print(f"  RMSE_v  rad={d_rad['rmse_v']:.6f}  tot={d_tot['rmse_v']:.6f}  m/s")
        print(f"  RMSE    rad={d_rad['rmse_total']:.6f}  tot={d_tot['rmse_total']:.6f}  m/s")
        print(f"  MAE_u   rad={d_rad['mae_u']:.6f}  tot={d_tot['mae_u']:.6f}  m/s")
        print(f"  MAE_v   rad={d_rad['mae_v']:.6f}  tot={d_tot['mae_v']:.6f}  m/s")
        print(f"  MAE     rad={d_rad['mae_total']:.6f}  tot={d_tot['mae_total']:.6f}  m/s")
        print(f"  STD_u   rad={d_rad['std_u']:.6f}  tot={d_tot['std_u']:.6f}  m/s")
        print(f"  STD_v   rad={d_rad['std_v']:.6f}  tot={d_tot['std_v']:.6f}  m/s")
        print(f"  STD     rad={d_rad['std_total']:.6f}  tot={d_tot['std_total']:.6f}  m/s")

# ------------------------------------------------------------------------------------
# SAVE OUTPUT FILES
# ------------------------------------------------------------------------------------

HEADER = ("i  "
          "N_rad  avg_u_rad  avg_v_rad  avg_speed_rad  rmse_u_rad  rmse_v_rad  rmse_total_rad  mae_u_rad  mae_v_rad  mae_total_rad  std_u_rad  std_v_rad  std_total_rad  "
          "N_tot  avg_u_tot  avg_v_tot  avg_speed_tot  rmse_u_tot  rmse_v_tot  rmse_total_tot  mae_u_tot  mae_v_tot  mae_total_tot  std_u_tot  std_v_tot  std_total_tot")

base_dir = "../data/january_2026/gaps_data"

output_small = os.path.join(base_dir, "gaps_stats_small_time_series.txt")
output_large = os.path.join(base_dir, "gaps_stats_large_time_series.txt")

np.savetxt(output_small, stats_small, fmt="%.6f", delimiter=" ", header=HEADER)
np.savetxt(output_large, stats_large, fmt="%.6f", delimiter=" ", header=HEADER)

print(f"\nSmall-gap statistics saved to: {output_small}")
print(f"Large-gap statistics saved to: {output_large}")
