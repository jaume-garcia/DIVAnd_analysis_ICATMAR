import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates

import seaborn as sns

# -----------------------------------------------------------------------
# Script: plot_rms_icatmar_grid_all_methods.py
# Purpose: Reads a time series of RMSE statistics comparing three HF radar
#          velocity reconstruction methods (DIVAnd-radials, DIVAnd-totals,
#          and Least Squares) against the Copernicus MEDSEA ground truth,
#          generates time series plots, and optionally prints summary
#          statistics to the console.
# -----------------------------------------------------------------------

# --- Global plot style ---
sns.set_style("darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")

plt.rcParams.update({
    "axes.facecolor":  "white",       # White axes background
    "figure.facecolor": "white",
    "grid.color":      "gray",
    "grid.linestyle":  "-",
    "grid.linewidth":  1.2,
    "axes.edgecolor":  "black",       # Black axis borders
    "axes.linewidth":  1.2,
    "xtick.color":     "black",
    "ytick.color":     "black",
    "axes.labelcolor": "black",
    "text.color":      "black"
})


def read_data(file):
    """
    Reads a whitespace-separated statistics file (no header) and assigns
    column names. Rows where all data columns are zero are filtered out.

    Parameters:
    -----------
    file : str
        Path to the input statistics file

    Returns:
    --------
    pd.DataFrame
        DataFrame with named columns and zero-rows removed
    """
    # Commented-out column set for extended file formats:
    # column_names = ['time', 'total_mse_LS', 'total_mse_rad', 'total_mse_tot',
    #                 'u_rmse_LS', 'u_rmse_rad', 'u_rmse_tot',
    #                 'v_rmse_LS', 'v_rmse_rad', 'v_rmse_tot',
    #                 'total_rmse_LS', 'total_rmse_rad', 'total_rmse_tot']
    column_names = ['time', 'total_rmse_LS', 'total_rmse_rad', 'total_rmse_tot']

    data = pd.read_csv(file, sep=r'\s+', header=None, names=column_names)

    # Remove rows where all data columns sum to zero (missing/invalid snapshots)
    data_cols = column_names[1:]   # All columns except 'time'
    data = data[data[data_cols].sum(axis=1) > 0]

    return data


def creating_dates(data, start_date_str='2025-02-05 00:00:00'):
    """
    Creates a 'date' column by offsetting the start date by the integer
    hour index stored in the 'time' column.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing a 'time' column with integer hour indices
    start_date_str : str
        Start date in 'YYYY-MM-DD HH:MM:SS' format

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional 'date' column
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    data['date'] = [start_date + timedelta(hours=int(h)) for h in data['time']]
    return data


def plot_u_rmse(data, output_file='../../figures/january_2026/statistics/u_rmse.png'):
    """Plots the RMSE time series for the east (U) velocity component."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['u_rmse_rad'], label='DIVAnd rad', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['u_rmse_tot'], label='DIVAnd tot', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['u_rmse_LS'],  label='LS',         linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (m/s)', fontsize=12)
    ax.set_title('RMSE U-component', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_v_rmse(data, output_file='../../figures/january_2026/statistics/v_rmse.png'):
    """Plots the RMSE time series for the north (V) velocity component."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['v_rmse_rad'], label='DIVAnd rad', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['v_rmse_tot'], label='DIVAnd tot', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['v_rmse_LS'],  label='LS',         linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (m/s)', fontsize=12)
    ax.set_title('RMSE V-component', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_total_rmse(data, output_file='../../figures/january_2026/statistics/total_rmse.png'):
    """
    Plots the total RMSE time series for all three reconstruction methods
    with daily major ticks and 12-hourly minor ticks on the x-axis.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(data['date'], data['total_rmse_LS'],  label='LS',      linewidth=4, color='orange')
    ax.plot(data['date'], data['total_rmse_tot'], label='Totals',  linewidth=4, color='blue')
    ax.plot(data['date'], data['total_rmse_rad'], label='Radials', linewidth=4, color='red')

    if not data.empty:
        # Determine the date range for axis limits
        date_min       = data['date'].min()
        date_max       = data['date'].max()
        date_min_floor = datetime(date_min.year, date_min.month, date_min.day)
        date_max_ceil  = datetime(date_max.year, date_max.month, date_max.day) + timedelta(days=1)

        # Major ticks every day at midnight; minor ticks every 12 hours
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))

        ax.set_xlim(date_min_floor, date_max_ceil)

        # Grid only on major ticks (daily)
        ax.grid(True, which='major', alpha=0.3)

        # Tick appearance
        ax.tick_params(axis='x', which='minor', length=8,  width=1,   color='black', direction='out')
        ax.tick_params(axis='x', which='major', length=12, width=1.5, color='black', direction='out', labelsize=24)
        ax.tick_params(axis='y', which='both',  labelsize=24, length=8, width=1.2,   color='black', direction='out')

        ax.tick_params(axis='x', which='minor', bottom=True)
        ax.tick_params(axis='x', which='major', bottom=True)

        ax.set_ylabel('RMSE (m/s)', fontsize=24, labelpad=14)
        ax.legend(fontsize=24)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor', fontsize=24)

        # Diagnostic: print tick positions
        major_ticks = ax.xaxis.get_majorticklocs()
        minor_ticks = ax.xaxis.get_minorticklocs()
        print(f"Major ticks (days):  {len(major_ticks)} ticks at: "
              f"{[mdates.num2date(t).strftime('%d-%H:%M') for t in major_ticks]}")
        print(f"Minor ticks (12 h):  {len(minor_ticks)} ticks")

        plt.tight_layout()
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")
    print(f"Saved: {output_file}")


def plot_total_mse(data, output_file='../../figures/january_2026/statistics/total_mse.png'):
    """Plots the total MSE time series for all three reconstruction methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['total_mse_rad'], label='DIVAnd rad', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['total_mse_tot'], label='DIVAnd tot', linewidth=1.5, alpha=0.8)
    ax.plot(data['date'], data['total_mse_LS'],  label='LS',         linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MSE Total', fontsize=12)
    ax.set_title('MSE Total', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def creating_plots(data):
    """Generates all configured plots and saves them to disk."""
    import os
    print("\nGenerating plots...")
    os.makedirs('./figures', exist_ok=True)

    # Uncomment individual functions to enable additional plots:
    # plot_u_rmse(data)
    # plot_v_rmse(data)
    plot_total_rmse(data)
    # plot_total_mse(data)

    print("\nPlots generated successfully.")


def print_statistics(data):
    """Prints basic summary statistics (mean, median, max) for all methods."""
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)

    print(f"\nNumber of records:  {len(data)}")
    print(f"Time range:         {data['time'].min()} - {data['time'].max()} hours")
    print(f"Date range:         {data['date'].min()} - {data['date'].max()}")

    # -- Total MSE --
    print("\n" + "-" * 80)
    print("TOTAL MSE:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean':<15} {'Median':<15} {'Max':<15}")
    print("-" * 80)
    for label, col in [('LS', 'total_mse_LS'), ('DIVAnd rad', 'total_mse_rad'), ('DIVAnd tot', 'total_mse_tot')]:
        print(f"{label:<20} {data[col].mean():.6e}  {data[col].median():.6e}  {data[col].max():.6e}")

    # -- RMSE U --
    print("\n" + "-" * 80)
    print("RMSE U-component:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean':<15} {'Median':<15} {'Max':<15}")
    print("-" * 80)
    for label, col in [('LS', 'u_rmse_LS'), ('DIVAnd rad', 'u_rmse_rad'), ('DIVAnd tot', 'u_rmse_tot')]:
        print(f"{label:<20} {data[col].mean():.6e}  {data[col].median():.6e}  {data[col].max():.6e}")

    # -- RMSE V --
    print("\n" + "-" * 80)
    print("RMSE V-component:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean':<15} {'Median':<15} {'Max':<15}")
    print("-" * 80)
    for label, col in [('LS', 'v_rmse_LS'), ('DIVAnd rad', 'v_rmse_rad'), ('DIVAnd tot', 'v_rmse_tot')]:
        print(f"{label:<20} {data[col].mean():.6e}  {data[col].median():.6e}  {data[col].max():.6e}")

    # -- Total RMSE --
    print("\n" + "-" * 80)
    print("TOTAL RMSE:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean':<15} {'Median':<15} {'Max':<15}")
    print("-" * 80)
    for label, col in [('LS', 'total_rmse_LS'), ('DIVAnd rad', 'total_rmse_rad'), ('DIVAnd tot', 'total_rmse_tot')]:
        print(f"{label:<20} {data[col].mean():.6e}  {data[col].median():.6e}  {data[col].max():.6e}")

    print("=" * 80)


# --- Main entry point ---
if __name__ == "__main__":

    # Input file path
    file_rms = "../../data/january_2026/stats_data/all_rms_grid_icatmar.txt"

    # Start date for the time axis
    date_ini = '2026-01-14 00:00:00'

    print("=" * 80)
    print("RMS ANALYSIS L1 - ICATMAR GRID")
    print("=" * 80)
    print(f"\nReading data from: {file_rms}")

    # Read and date-stamp the data
    data = read_data(file_rms)
    data = creating_dates(data, date_ini)

    # Print statistics (optional — uncomment to enable)
    # print_statistics(data)

    # Generate and save all plots
    creating_plots(data)

    print("\nProcess completed successfully.")
    print("=" * 80)
