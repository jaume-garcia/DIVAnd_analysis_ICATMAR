import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
from netCDF4 import Dataset
import os

# -----------------------------------------------------------------------
# Script: plot_time_series_vel_points.py
# Purpose: Generates two time series figures for the January 2026 10-day
#          period:
#            1. Domain-averaged velocity magnitude for Copernicus MEDSEA
#               and the three reconstruction methods (DIVAnd-rad, DIVAnd-tot,
#               Least Squares).
#            2. Spatial standard deviation of the velocity magnitude
#               difference (method vs Copernicus), with the number of valid
#               grid points on a secondary y-axis.
# -----------------------------------------------------------------------

# --- Global plot style ---
plt.style.use("seaborn-v0_8-darkgrid")

plt.rcParams.update({
    "axes.facecolor":  "white",
    "figure.facecolor": "white",
    "grid.color":      "gray",
    "grid.linestyle":  "-",
    "grid.linewidth":  1.2,
    "axes.edgecolor":  "black",
    "axes.linewidth":  1.2,
    "xtick.color":     "black",
    "ytick.color":     "black",
    "axes.labelcolor": "black",
    "text.color":      "black"
})



# --- Colour coding by reconstruction method ---
COLOR_COP    = "black"
COLOR_COPTEO = "purple"
COLOR_RAD    = "red"
COLOR_TOT    = "blue"
COLOR_LS     = "orange"
COLOR_NPTS   = "black"


# --- Helper functions ---

def _format_time_axis(ax, fechas):
    """
    Apply a consistent time format to the X-axis.
	Major ticks for each day (label = day number),
	minor ticks for each hour.
    """
    if len(fechas) == 0:
        return

    fecha_min = pd.Series(fechas).min()
    fecha_max = pd.Series(fechas).max()
    fecha_min_floor = datetime(fecha_min.year, fecha_min.month, fecha_min.day)
    fecha_max_ceil  = datetime(fecha_max.year, fecha_max.month, fecha_max.day) + timedelta(days=1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.set_xlim(fecha_min_floor, fecha_max_ceil)

    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(axis='x', which='minor', length=8,  width=1,   color='black', direction='out', bottom=True)
    ax.tick_params(axis='x', which='major', length=12, width=1.5, color='black', direction='out', labelsize=18)
    ax.tick_params(axis='y', which='both',  labelsize=18, length=8, width=1.2,  color='black', direction='out')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')


def _save(fig, output_file):
    """Saves the figure to disk and closes it."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(archivo_salida) else '.', exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")


# --- Date vector ---

start_date = datetime(2026, 1, 14, 0, 0, 0)
dates = [start_date + timedelta(hours=h) for h in range(264)]


# --- 1. Velocity magnitude time series ---

column_names = ['tiempo', 'avg_vel_cop', 'avg_vel_cop_teo', 'avg_vel_rad', 'avg_vel_tot', 'avg_vel_LS']

data_avg = pd.read_csv(
    "../../data/january_2026/velocity_magnitude_times_series.txt",
    sep=r'\s+', header=None, names=column_names
)

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(dates, data_avg['avg_vel_cop'],     label='MEDSEA',                linewidth=4, color=COLOR_COP)
ax.plot(dates, data_avg['avg_vel_cop_teo'], label='MEDSEA theo. coverage', linewidth=4, color=COLOR_COPTEO, linestyle='--')
ax.plot(dates, data_avg['avg_vel_rad'],     label='DIVAnd rad',            linewidth=1, color=COLOR_RAD)
ax.plot(dates, data_avg['avg_vel_tot'],     label='DIVAnd tot',            linewidth=1, color=COLOR_TOT)
ax.plot(dates, data_avg['avg_vel_LS'],      label='LS',                    linewidth=1, color=COLOR_LS)

ax.set_ylabel('Average velocity module (m/s)', fontsize=18, labelpad=12)
ax.set_title('Time series of spatial average velocity module', fontsize=20, fontweight='bold')
ax.legend(fontsize=16)
_format_time_axis(ax, dates)
_save(fig, '../../figures/january_2026/statistics/times_series_spatial_avg_vel.png')


# --- 2. Standard deviation of velocity magnitude difference ---

column_names = ['tiempo', 'n_valid_points', 'avg_std_rad', 'avg_std_tot', 'avg_std_LS']

data_std = pd.read_csv(
    "../../figures/january_2026/statistics/standard_deviation_diff_times_series.txt",
    sep=r'\s+', header=None, names=column_names
)

fig, ax_left = plt.subplots(figsize=(16, 7))
ax_right = ax_left.twinx()

# Right axis: number of valid points
ax_right.plot(dates, data_std['n_valid_points'],
              label='Valid points', color=COLOR_NPTS, linewidth=1.5, zorder=0)
ax_right.set_ylabel('Number of valid points', fontsize=18, labelpad=12)
ax_right.tick_params(axis='y', labelcolor='black', labelsize=18,
                     length=8, width=1.2, direction='out')
ax_right.set_ylim(top=10000)

# Left axis: standard deviation time series
ax_left.plot(dates, data_std['avg_std_rad'], label='DIVAnd rad', linewidth=2.5, color=COLOR_RAD,  zorder=1)
ax_left.plot(dates, data_std['avg_std_tot'], label='DIVAnd tot', linewidth=2.5, color=COLOR_TOT,  zorder=2)
ax_left.plot(dates, data_std['avg_std_LS'],  label='LS',         linewidth=2.5, color=COLOR_LS,   zorder=3)

ax_left.set_ylabel('Standard deviation (m/s)', fontsize=18, labelpad=12)
ax_left.tick_params(axis='y', labelsize=18, length=8, width=1.2, direction='out')
ax_left.set_ylim(bottom=0.01)
ax_left.legend(fontsize=16, loc='upper right')

ax_left.set_title('Time series of spatial std of velocity magnitude difference',
                  fontsize=20, fontweight='bold')

_format_time_axis(ax_left, dates)
_save(fig, '../../figures/january_2026/statistics/times_series_spatial_avg_std_diff.png')
