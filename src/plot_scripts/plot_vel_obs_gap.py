import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import os


# -----------------------------------------------------------------------
# Script: plot_vel_obs_gap.py
# Purpose: Plots velocity time series at a fixed geographic point located
#          inside a large HF radar data gap. Three figures are generated:
#          the east (U) component, the north (V) component, and the velocity
#          magnitude. Each figure compares the Copernicus MEDSEA ground
#          truth against the three reconstruction methods. Shaded vertical
#          spans indicate time steps with no available LS data (NaN).
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
COLOR_COP = "black"
COLOR_COP_TOT = "grey"
COLOR_RAD = "red"
COLOR_TOT = "blue"
COLOR_LS  = "orange"


# --- Helper functions ---

def _format_time_axis(ax, fechas):
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


def _draw_gaps(ax, dates, mask_series, color='lightcoral', alpha=0.3, label='No data'):
    """
    Plot vertical bars where mask_series is NaN.

    Parameters
    ----------
    ax          : Matplotlib Axes object.
    dates       : List/array of datetime objects (same length as mask_series).
    mask_series : pd.Series or array — reference column (NaN = no data).
    color       : Shading colour (default: 'lightcoral').
    alpha       : Transparency (default: 0.3).
    label       : Legend label (applied to the first span only).
    """
    dates = list(dates)
    nan_mask = np.isnan(np.array(mask_series, dtype=float))
    half_step = (dates[1] - dates[0]) / 2 if len(dates) > 1 else timedelta(hours=0)

    label_used = False
    i = 0
    while i < len(dates):
        if nan_mask[i]:
            start = dates[i]
            while i < len(dates) and nan_mask[i]:
                i += 1
            end = dates[i - 1]
            ax.axvspan(
                start - half_step,
                end   + half_step,
                color=color,
                alpha=alpha,
                zorder=0,
                label=label if not label_used else '_nolegend_'
            )
            label_used = True
        else:
            i += 1


def _save(fig, archivo_salida):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(archivo_salida) else '.', exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")


# --- Date vector ---

start_date = datetime(2026, 1, 14, 0, 0, 0)
dates = [start_date + timedelta(hours=h) for h in range(264)]


# --- Data loading ---


# --- Observation gap data ---

column_names = [
    'tiempo',
    'cop_interp_u', 'interpolated_u_rad', 'interpolated_u_tot', 'interpolated_u_LS',
    'cop_interp_v', 'interpolated_v_rad', 'interpolated_v_tot', 'interpolated_v_LS',
    'vel_mag_cop',  'vel_mag_interp_rad', 'vel_mag_interp_tot', 'vel_mag_interp_LS'
]

data = pd.read_csv(
    "../../data/january_2026/gaps_data/stats_vel_obs_point_large_gap.txt",
    sep=r'\s+', header=None, names=column_names
)


lon_point = 3.350140 # 3.491500 #
lat_point = 41.502102  # 41.529099#


# --- Domain-averaged Copernicus velocity ---

column_names = ['tiempo', 'avg_vel_cop', 'avg_vel_cop_teo', 'avg_vel_rad', 'avg_vel_tot', 'avg_vel_LS']

data_avg = pd.read_csv("../../data/january_2026/velocity_magnitude_times_series.txt",sep=r'\s+', header=None, names=column_names)


# --- 1. East (U) velocity component ---

fig, ax = plt.subplots(figsize=(16, 7))

_draw_gaps(ax, dates, data['interpolated_u_LS'], color='lightcoral', alpha=0.3, label='No data')

ax.plot(dates, data['cop_interp_u'],       label='MEDSEA',     linewidth=2, color=COLOR_COP)
ax.plot(dates, data['interpolated_u_rad'], label='DIVAnd rad',  linewidth=1, color=COLOR_RAD)
ax.plot(dates, data['interpolated_u_tot'], label='DIVAnd tot',  linewidth=1, color=COLOR_TOT)
ax.plot(dates, data['interpolated_u_LS'],  label='LS',          linewidth=2, color=COLOR_LS)

ax.set_ylabel('U-component (m/s)', fontsize=18, labelpad=12)
ax.set_title('Large gap obs', fontsize=20, fontweight='bold')
ax.legend(fontsize=16)
ax.text(0.01, 0.97, f"lon = {lon_point:.4f}°\nlat = {lat_point:.4f}°",
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.2))
_format_time_axis(ax, dates)
_save(fig, '../../figures/january_2026/gaps_by_size/obs_gap_large/times_series_vel_obs_point_u.png')


# --- 2. North (V) velocity component ---

fig, ax = plt.subplots(figsize=(16, 7))

_draw_gaps(ax, dates, data['interpolated_v_LS'], color='lightcoral', alpha=0.3, label='No data')

ax.plot(dates, data['cop_interp_v'],       label='MEDSEA',     linewidth=2, color=COLOR_COP)
ax.plot(dates, data['interpolated_v_rad'], label='DIVAnd rad',  linewidth=1, color=COLOR_RAD)
ax.plot(dates, data['interpolated_v_tot'], label='DIVAnd tot',  linewidth=1, color=COLOR_TOT)
ax.plot(dates, data['interpolated_v_LS'],  label='LS',          linewidth=2, color=COLOR_LS)

ax.set_ylabel('V-component (m/s)', fontsize=18, labelpad=12)
ax.set_title('Large gap obs', fontsize=20, fontweight='bold')
ax.legend(fontsize=16)
ax.text(0.01, 0.97, f"lon = {lon_point:.4f}°\nlat = {lat_point:.4f}°",
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.2))
_format_time_axis(ax, dates)
_save(fig, '../../figures/january_2026/gaps_by_size/obs_gap_large/times_series_vel_obs_point_v.png')


# --- 3. Velocity magnitude ---

fig, ax = plt.subplots(figsize=(16, 7))

_draw_gaps(ax, dates, data['vel_mag_interp_LS'], color='lightcoral', alpha=0.3, label='No data')

ax.plot(dates, data['vel_mag_cop'], label='MEDSEA',     linewidth=2, color=COLOR_COP)
ax.plot(dates, data_avg['avg_vel_cop'],  label='MEDSEA total',linewidth=2, linestyle='--',color=COLOR_COP_TOT)
ax.plot(dates, data['vel_mag_interp_rad'], label='DIVAnd rad',  linewidth=1, color=COLOR_RAD)
ax.plot(dates, data['vel_mag_interp_tot'], label='DIVAnd tot',  linewidth=1, color=COLOR_TOT)
ax.plot(dates, data['vel_mag_interp_LS'],  label='LS',          linewidth=2, color=COLOR_LS)

ax.set_ylabel('Velocity magnitude (m/s)', fontsize=18, labelpad=12)
ax.set_title('Large gap obs', fontsize=20, fontweight='bold')
ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=4, frameon=True, framealpha=1, edgecolor='black')
ax.text(0.01, 0.97, f"lon = {lon_point:.4f}°\nlat = {lat_point:.4f}°",
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.2))
_format_time_axis(ax, dates)
ax.set_ylim(bottom=0, top=0.9)
_save(fig, '../../figures/january_2026/gaps_by_size/obs_gap_large/times_series_vel_obs_point_mag.png')
