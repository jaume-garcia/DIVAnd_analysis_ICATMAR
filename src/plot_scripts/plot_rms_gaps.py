import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os

import seaborn as sns


# -----------------------------------------------------------------------
# Script: plot_rms_gaps.py
# Purpose: Reads a time series of error statistics comparing DIVAnd-radials
#          and DIVAnd-totals reconstructions against the Copernicus MEDSEA
#          model at gap locations, and generates time series plots of average
#          bias, RMSE, MAE, and STD for the U/V velocity components and the
#          velocity magnitude. A 2x2 RMSE/MAE panel and a 1x2 STD panel are
#          also produced for compact comparison.
# -----------------------------------------------------------------------

# --- Global plot style ---
sns.set_style("darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")

plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "gray",
    "grid.linestyle": "-",
    "grid.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black"
})

# --- Colour coding by reconstruction method ---
COLOR_RAD = "red"
COLOR_TOT = "blue"


def read_data(file):
    """
    Lee el archivo de datos sin encabezados y asigna nombres a las columnas.

    Formato esperado (una fila por timestep):
        i  avg_u_rad  avg_v_rad  avg_mag_rad
           avg_u_tot  avg_v_tot  avg_mag_tot
           rmse_u_rad  rmse_v_rad  total_rmse_rad
           rmse_u_tot  rmse_v_tot  total_rmse_tot
           mae_u_rad   mae_v_rad   total_mae_rad
           mae_u_tot   mae_v_tot   total_mae_tot
           std_u_rad   std_v_rad   std_mag_rad
           std_u_tot   std_v_tot   std_mag_tot

    Args:
        archivo (str): Ruta al archivo de datos.

    Returns:
        pd.DataFrame: DataFrame con los datos leídos y filtrados.
    """
    column_names = [
        'tiempo',
        'avg_u_rad',   'avg_v_rad',   'avg_mag_rad',
        'avg_u_tot',   'avg_v_tot',   'avg_mag_tot',
        'rmse_u_rad',  'rmse_v_rad',  'total_rmse_rad',
        'rmse_u_tot',  'rmse_v_tot',  'total_rmse_tot',
        'mae_u_rad',   'mae_v_rad',   'total_mae_rad',
        'mae_u_tot',   'mae_v_tot',   'total_mae_tot',
        'std_u_rad',   'std_v_rad',   'std_mag_rad',
        'std_u_tot',   'std_v_tot',   'std_mag_tot',
    ]

    data = pd.read_csv(file, sep=r'\s+', header=None, names=column_names)

    # Remove rows where all data columns sum to zero (missing/invalid snapshots)
    data_cols = column_names[1:]
    data = data[data[data_cols].sum(axis=1) > 0].reset_index(drop=True)

    return data


def creating_dates(data, start_date_str='2026-01-14 00:00:00'):
    """
    Crea columna de fechas a partir de la columna 'tiempo' (índice horario entero).

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        fecha_inicio_str (str): Fecha de inicio en formato 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        pd.DataFrame: DataFrame con columna 'fecha' añadida.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    data['date'] = [start_date + timedelta(hours=int(h)) for h in data['time']]
    return data


# --- Time-axis formatting helpers ---

def _format_time_axis(ax, data):
    """Applies consistent time formatting to the X axis.
    Major ticks every day; minor ticks every 12 hours."""
    if data.empty:
        return

    fecha_min = data['date'].min()
    fecha_max = data['date'].max()
    fecha_min_floor = datetime(fecha_min.year, fecha_min.month, fecha_min.day)
    fecha_max_ceil  = datetime(fecha_max.year, fecha_max.month, fecha_max.day) + timedelta(days=1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.set_xlim(fecha_min_floor, fecha_max_ceil)

    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(axis='x', which='minor', length=8,  width=1,   color='black', direction='out', bottom=True)
    ax.tick_params(axis='x', which='major', length=12, width=1.5, color='black', direction='out', labelsize=18)
    ax.tick_params(axis='y', which='both',  labelsize=18, length=8, width=1.2, color='black', direction='out')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')


def _save(fig, output_file):
    """Saves the figure to disk and closes it."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(archivo_salida) else '.', exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")


# --- Individual plot functions ---

def plot_avg_u(data, output_file='../../figures/january_2026/statistics/avg_u.png'):
    """Plots the average bias for the east (U) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Average bias U (m/s)', fontsize=18, labelpad=12)
    ax.set_title('Average bias U-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_avg_v(data, output_file='../../figures/january_2026/statistics/avg_v.png'):
    """Plots the average bias for the north (V) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Average bias V (m/s)', fontsize=18, labelpad=12)
    ax.set_title('Average bias V-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_avg_mag(data, output_file='../../figures/january_2026/statistics/avg_mag.png'):
    """Plots the average bias for the velocity magnitude."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_mag_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_mag_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('Average bias velocity magnitude (m/s)', fontsize=18, labelpad=12)
    ax.set_title('Average bias velocity magnitude', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_rmse_u(data, output_file='../../figures/january_2026/statistics/rmse_u.png'):
    """Plots the RMSE time series for the east (U) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['rmse_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['rmse_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('RMSE U (m/s)', fontsize=18, labelpad=12)
    ax.set_title('RMSE U-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_rmse_v(data, output_file='../../figures/january_2026/statistics/rmse_v.png'):
    """Plots the RMSE time series for the north (V) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['rmse_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['rmse_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('RMSE V (m/s)', fontsize=18, labelpad=12)
    ax.set_title('RMSE V-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_total_rmse(data, output_file='../../figures/january_2026/statistics/total_rmse.png'):
    """Plots the total RMSE time series."""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['total_rmse_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['total_rmse_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('RMSE (m/s)', fontsize=24, labelpad=14)
    ax.set_title('Total RMSE', fontsize=26, fontweight='bold')
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_mae_u(data, output_file='../../figures/january_2026/statistics/mae_u.png'):
    """Plots the MAE time series for the east (U) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['mae_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['mae_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('MAE U (m/s)', fontsize=18, labelpad=12)
    ax.set_title('MAE U-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_mae_v(data, output_file='../../figures/january_2026/statistics/mae_v.png'):
    """Plots the MAE time series for the north (V) velocity component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['mae_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['mae_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('MAE V (m/s)', fontsize=18, labelpad=12)
    ax.set_title('MAE V-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_total_mae(data, output_file='../../figures/january_2026/statistics/total_mae.png'):
    """Plots the total MAE time series."""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['total_mae_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['total_mae_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('MAE (m/s)', fontsize=24, labelpad=14)
    ax.set_title('Total MAE', fontsize=26, fontweight='bold')
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_panel_rmse_mae(data, output_file='../../figures/january_2026/statistics/panel_rmse_mae.png'):
    """
    Two-by-two panel figure: RMSE-U, RMSE-V, MAE-U, MAE-V per component.
    Useful for a quick at-a-glance comparison across all metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharex=True)
    fig.suptitle('RMSE & MAE by Component', fontsize=22, fontweight='bold', y=1.01)

    pairs = [
        (axes[0, 0], 'rmse_u_rad', 'rmse_u_tot', 'RMSE U (m/s)'),
        (axes[0, 1], 'rmse_v_rad', 'rmse_v_tot', 'RMSE V (m/s)'),
        (axes[1, 0], 'mae_u_rad',  'mae_u_tot',  'MAE U (m/s)'),
        (axes[1, 1], 'mae_v_rad',  'mae_v_tot',  'MAE V (m/s)'),
    ]

    for ax, col_rad, col_tot, ylabel in pairs:
        ax.plot(data['date'], data[col_rad], label='Radials', linewidth=2, color=COLOR_RAD)
        ax.plot(data['date'], data[col_tot], label='Totals',  linewidth=2, color=COLOR_TOT)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        _format_time_axis(ax, data)

    plt.tight_layout()
    _save(fig, archivo_salida)


# --- Standard deviation plot functions ---

def plot_std_u(data, output_file='../../figures/january_2026/statistics/std_u.png'):
    """Plots the standard deviation of the velocity difference for the U component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['std_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['std_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('STD U (m/s)', fontsize=18, labelpad=12)
    ax.set_title('STD of difference U-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_std_v(data, output_file='../../figures/january_2026/statistics/std_v.png'):
    """Plots the standard deviation of the velocity difference for the V component."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['std_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['std_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('STD V (m/s)', fontsize=18, labelpad=12)
    ax.set_title('STD of difference V-component', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_std_mag(data, output_file='../../figures/january_2026/statistics/std_mag.png'):
    """Plots the standard deviation of the velocity magnitude difference."""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['std_mag_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['std_mag_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('STD velocity magnitude (m/s)', fontsize=24, labelpad=14)
    ax.set_title('STD of difference - velocity magnitude', fontsize=26, fontweight='bold')
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)


def plot_panel_std(data, output_file='../../figures/january_2026/statistics/panel_std.png'):
    """
    One-by-two panel showing STD-U and STD-V side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 7), sharex=True)
    fig.suptitle('STD of differences by Component', fontsize=22, fontweight='bold', y=1.02)

    pairs = [
        (axes[0], 'std_u_rad', 'std_u_tot', 'STD U (m/s)'),
        (axes[1], 'std_v_rad', 'std_v_tot', 'STD V (m/s)'),
    ]

    for ax, col_rad, col_tot, ylabel in pairs:
        ax.plot(data['date'], data[col_rad], label='Radials', linewidth=2, color=COLOR_RAD)
        ax.plot(data['date'], data[col_tot], label='Totals',  linewidth=2, color=COLOR_TOT)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        _format_time_axis(ax, data)

    plt.tight_layout()
    _save(fig, archivo_salida)


# --- Summary statistics ---

def print_statistics(data):
    """Prints basic summary statistics (mean, median, max) to the console."""
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    print(f"\nNumber of records:  {len(data)}")
    print(f"Time range:         {data['time'].min()} - {data['time'].max()} hours")
    print(f"Date range:         {data['date'].min()} - {data['date'].max()}")

    sections = [
        ("AVG U",         'avg_u_rad',      'avg_u_tot'),
        ("AVG V",         'avg_v_rad',      'avg_v_tot'),
        ("AVG MAG",       'avg_mag_rad',    'avg_mag_tot'),
        ("RMSE U",        'rmse_u_rad',     'rmse_u_tot'),
        ("RMSE V",        'rmse_v_rad',     'rmse_v_tot'),
        ("RMSE TOTAL",    'total_rmse_rad', 'total_rmse_tot'),
        ("MAE U",         'mae_u_rad',      'mae_u_tot'),
        ("MAE V",         'mae_v_rad',      'mae_v_tot'),
        ("MAE TOTAL",     'total_mae_rad',  'total_mae_tot'),
        ("STD U",         'std_u_rad',      'std_u_tot'),
        ("STD V",         'std_v_rad',      'std_v_tot'),
        ("STD MAG",       'std_mag_rad',    'std_mag_tot'),
    ]

    hdr = f"{'Metric':<14} {'Method':<10} {'Mean':>14} {'Median':>14} {'Max':>14}"
    sep = "-" * 70

    for title, col_rad, col_tot in sections:
        print(f"\n{sep}")
        print(f" {title}")
        print(sep)
        print(hdr)
        print(sep)
        for label, col in [("Radials", col_rad), ("Totals", col_tot)]:
            s = data[col]
            print(f"{title:<14} {label:<10} {s.mean():>14.6e} {s.median():>14.6e} {s.max():>14.6e}")

    print("\n" + "=" * 80)


# --- Plot generation ---

def create_all_plots(data, output_dir='../../figures/january_2026/statistics'):
    """Creates all configured plots and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating plots...")

    def path(nombre):
        return os.path.join(output_dir, nombre)

    plot_avg_u(data,           path('avg_u.png'))
    plot_avg_v(data,           path('avg_v.png'))
    plot_avg_mag(data,         path('avg_mag.png'))
    plot_rmse_u(data,          path('rmse_u.png'))
    plot_rmse_v(data,          path('rmse_v.png'))
    plot_total_rmse(data,      path('total_rmse.png'))
    plot_mae_u(data,           path('mae_u.png'))
    plot_mae_v(data,           path('mae_v.png'))
    plot_total_mae(data,       path('total_mae.png'))
    plot_panel_rmse_mae(data,  path('panel_rmse_mae.png'))
    plot_std_u(data,           path('std_u.png'))
    plot_std_v(data,           path('std_v.png'))
    plot_std_mag(data,         path('std_mag.png'))
    plot_panel_std(data,       path('panel_std.png'))

    print("\nPlots generated successfully.")


# --- Main entry point ---

if __name__ == "__main__":
        # --- Configuration ---
    file_metrics = "../../data/january_2026/gaps_data/gaps_avg_time_series.txt"
    date_ini     = '2026-01-14 00:00:00'
    output_dir   = '../../figures/january_2026/gaps'

    print("=" * 80)
    print("METRICS ANALYSIS - RADIALS vs TOTALS")
    print("=" * 80)
    print(f"\nLeyendo datos desde: {file_metrics}")

    # Leer datos
    data = read_data(file_metrics)

    # Crear columna de fechas
    data = creating_dates(data, date_ini)

    # Imprimir estadísticas (opcional)
    print_statistics(data)

    # Crear gráficos
    create_all_plots(data, output_dir=output_dir)

    print("\n✓ Process completed.")
    print("=" * 80)
