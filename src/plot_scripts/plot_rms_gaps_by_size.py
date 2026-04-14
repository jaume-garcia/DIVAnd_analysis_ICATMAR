import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os
 
import seaborn as sns
 
# -----------------------------------------------------------------------
# Script: plot_rms_gaps_by_size.py
# Purpose: Reads statistical error time series for two gap size classes
#          (small: < 4 consecutive missing observations, large: >= 4) and
#          generates individual, comparative, and panel figures of average
#          bias, RMSE, MAE, and STD for DIVAnd-radials vs DIVAnd-totals
#          reconstructions. Also produces a horizontal RMSE comparison panel
#          and a vertical panel combining gap-point counts with RMSE.
# -----------------------------------------------------------------------

# --- Global plot style ---
sns.set_style("darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")
 
plt.rcParams.update({
    "axes.facecolor":  "white",
    "figure.facecolor":"white",
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
COLOR_RAD = "red"
COLOR_TOT = "blue"
 
# --- Colour coding by gap size ---
#   Used in small vs large comparison panels
COLOR_SMALL = "royalblue"
COLOR_LARGE = "tomato"
 
# --- Fixed Y-axis limits for error metrics (m/s) ---
#   Applied to RMSE, MAE y STD. N_rad and bias (avg) remain with autoscale.
YLIM_METRIC = (0.1, 1.0)   # (m/s)
 
 
# --- Data loading ---
 
def read_data(archivo):
    """
    Read one of the files containing statistics by gap size.
 
    Expected format (created by loop_gap_filling_divand_by_size.py):
        
        N_rad  avg_u_rad  avg_v_rad  avg_mag_rad
               rmse_u_rad rmse_v_rad rmse_tot_rad
               mae_u_rad  mae_v_rad  mae_tot_rad
        N_tot  avg_u_tot  avg_v_tot  avg_mag_tot
               rmse_u_tot rmse_v_tot rmse_tot_tot
               mae_u_tot  mae_v_tot  mae_tot_tot
    """
    column_names = [
        'tiempo',
        # --- radials ---
        'N_rad',
        'avg_u_rad',    'avg_v_rad',    'avg_mag_rad',
        'rmse_u_rad',   'rmse_v_rad',   'total_rmse_rad',
        'mae_u_rad',    'mae_v_rad',    'total_mae_rad',
        'std_u_rad',    'std_v_rad',    'std_tot_rad',
        # --- totals ---
        'N_tot',
        'avg_u_tot',    'avg_v_tot',    'avg_mag_tot',
        'rmse_u_tot',   'rmse_v_tot',   'total_rmse_tot',
        'mae_u_tot',    'mae_v_tot',    'total_mae_tot',
        'std_u_tot',    'std_v_tot',    'std_tot_tot',
    ]
 
    data = pd.read_csv(archivo, sep=r'\s+', comment='#',
                       header=None, names=column_names)
 
    # Filtering rows where all data columns are zero or NaN
    data_cols = column_names[1:]
    data = data[data[data_cols].sum(axis=1) > 0].reset_index(drop=True)
 
    return data
 
 
def creating_dates(data, start_date_str='2026-01-14 00:00:00'):
    """Creates a 'date' column from the integer hour index in the 'time' column."""
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    data['date'] = [start_date + timedelta(hours=int(h)) for h in data['time']]
    return data
 
 
# --- Time-axis formatting helpers ---
 
def _format_time_axis(ax, data):
    """Applies consistent time formatting to the X axis."""
    if data.empty:
        return
 
    fecha_min = data['date'].min()
    fecha_max = data['date'].max()
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
 
 
# --- Individual plots per gap size ---
 
def plot_avg_u(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Average bias U (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'Average bias U-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_avg_v(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Average bias V (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'Average bias V-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_avg_mag(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['avg_mag_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['avg_mag_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Average bias velocity magnitude (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'Average bias velocity magnitude  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_rmse_u(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['rmse_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['rmse_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('RMSE U (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'RMSE U-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_rmse_v(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['rmse_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['rmse_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('RMSE V (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'RMSE V-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_total_rmse(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['total_rmse_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['total_rmse_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('RMSE (m/s)', fontsize=24, labelpad=14)
    ax.set_title(f'Total RMSE  [{gap_label}]', fontsize=26, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_mae_u(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['mae_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['mae_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('MAE U (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'MAE U-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_mae_v(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['mae_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['mae_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('MAE V (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'MAE V-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_total_mae(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['total_mae_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['total_mae_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('MAE (m/s)', fontsize=24, labelpad=14)
    ax.set_title(f'Total MAE  [{gap_label}]', fontsize=26, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_std_u(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['std_u_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['std_u_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('STD U (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'STD U-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_std_v(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['std_v_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['std_v_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT)
    ax.set_ylabel('STD V (m/s)', fontsize=18, labelpad=12)
    ax.set_title(f'STD V-component  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_total_std(data, gap_label, archivo_salida):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(data['date'], data['std_tot_rad'], label='Radials', linewidth=4, color=COLOR_RAD)
    ax.plot(data['date'], data['std_tot_tot'], label='Totals',  linewidth=4, color=COLOR_TOT)
    ax.set_ylabel('STD (m/s)', fontsize=24, labelpad=14)
    ax.set_title(f'Total STD  [{gap_label}]', fontsize=26, fontweight='bold')
    ax.set_ylim(YLIM_METRIC)
    ax.legend(fontsize=24)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_N(data, gap_label, archivo_salida):
    """Plots the number of gap pixels used at each time step."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data['date'], data['N_rad'], label='Radials', linewidth=2.5, color=COLOR_RAD)
    ax.plot(data['date'], data['N_tot'], label='Totals',  linewidth=2.5, color=COLOR_TOT, linestyle='--')
    ax.set_ylabel('N gap pixels', fontsize=18, labelpad=12)
    ax.set_title(f'Number of gap pixels used  [{gap_label}]', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    _format_time_axis(ax, data)
    _save(fig, archivo_salida)
 
 
def plot_panel_rmse_mae(data, gap_label, archivo_salida):
    """Panel 3×4: RMSE / MAE / STD by components (U-rad, V-rad, U-tot, V-tot)."""
    fig, axes = plt.subplots(3, 4, figsize=(28, 16), sharex=False)
    fig.suptitle(f'RMSE, MAE & STD by Component  [{gap_label}]',
                 fontsize=22, fontweight='bold', y=1.01)
 
    layout = [
        # row 0: RMSE
        (axes[0, 0], 'rmse_u_rad',  'rmse_u_tot',  'RMSE U (m/s)'),
        (axes[0, 1], 'rmse_v_rad',  'rmse_v_tot',  'RMSE V (m/s)'),
        (axes[0, 2], 'total_rmse_rad', 'total_rmse_tot', 'Total RMSE (m/s)'),
        (axes[0, 3], 'N_rad',        'N_tot',       'N pixels'),
        # row 1: MAE
        (axes[1, 0], 'mae_u_rad',   'mae_u_tot',   'MAE U (m/s)'),
        (axes[1, 1], 'mae_v_rad',   'mae_v_tot',   'MAE V (m/s)'),
        (axes[1, 2], 'total_mae_rad', 'total_mae_tot', 'Total MAE (m/s)'),
        (axes[1, 3], 'avg_mag_rad', 'avg_mag_tot', 'Avg |vel| (m/s)'),
        # row 2: STD
        (axes[2, 0], 'std_u_rad',   'std_u_tot',   'STD U (m/s)'),
        (axes[2, 1], 'std_v_rad',   'std_v_tot',   'STD V (m/s)'),
        (axes[2, 2], 'std_tot_rad', 'std_tot_tot', 'Total STD (m/s)'),
        (axes[2, 3], 'avg_u_rad',   'avg_u_tot',   'Avg bias U (m/s)'),
    ]
    # Columns that should NOT have the fixed metric ylim
    _NO_YLIM = {'N_rad', 'N_tot', 'avg_u_rad', 'avg_u_tot', 'avg_v_rad', 'avg_v_tot',
                'avg_mag_rad', 'avg_mag_tot'}
 
    for ax, col_rad, col_tot, ylabel in layout:
        ax.plot(data['date'], data[col_rad], label='Radials', linewidth=2, color=COLOR_RAD)
        ax.plot(data['date'], data[col_tot], label='Totals',  linewidth=2, color=COLOR_TOT)
        ax.set_ylabel(ylabel, fontsize=12)
        if col_rad not in _NO_YLIM and col_tot not in _NO_YLIM:
            ax.set_ylim(YLIM_METRIC)
        ax.legend(fontsize=10)
        _format_time_axis(ax, data)
 
    plt.tight_layout()
    _save(fig, archivo_salida)
 
 
# --- Vertical panel: N points + RMSE by gap size ---
 
def plot_panel_N_and_rmse(data_small, data_large, archivo_salida):
    """
    Panel vertical con 3 subplots:
      1. N_rad a lo largo del tiempo para small gaps y large gaps (juntos)
      2. Total RMSE para small gaps — lineas DIVAnd Radials vs DIVAnd Totals
      3. Total RMSE para large gaps — lineas DIVAnd Radials vs DIVAnd Totals
    """
    fig, axes = plt.subplots(3, 1, figsize=(22, 18), sharex=False)
    fig.suptitle('N gap points & Total RMSE — DIVAnd Radials vs Totals',
                 fontsize=22, fontweight='bold', y=1.01)
 
    # -- Panel 0: number of gap points (small and large together) -------------
    ax0 = axes[0]
    data_ref = data_small if not data_small.empty else data_large
    ax0.plot(data_small['fecha'], data_small['N_rad'],
             label='Small gaps (<4)',  linewidth=2.5,marker='o',markersize=4, color=COLOR_SMALL)
    ax0.plot(data_large['fecha'], data_large['N_rad'],
             label='Large gaps (>=4)', linewidth=2.5, marker='o',markersize=4,color=COLOR_LARGE)
    ax0.set_ylabel('N gap points', fontsize=16, labelpad=12)
    ax0.set_title('Number of gap points over time', fontsize=18, fontweight='bold')
    ax0.legend(fontsize=14)
    _format_time_axis(ax0, data_ref)
 
    # -- Panel 1: Total RMSE - Small gaps (Radials vs Totals) -----------------
    ax1 = axes[1]
    ax1.plot(data_small['fecha'], data_small['total_rmse_rad'],
             label='DIVAnd Radials', linewidth=2.5,marker='o',markersize=4, color=COLOR_RAD)
    ax1.plot(data_small['fecha'], data_small['total_rmse_tot'],
             label='DIVAnd Totals',  linewidth=2.5,marker='o',markersize=4, color=COLOR_TOT)
    ax1.set_ylabel('Total RMSE (m/s)', fontsize=16, labelpad=12)
    ax1.set_title('Total RMSE - Small gaps (<4)', fontsize=18, fontweight='bold')
    ax1.set_ylim(YLIM_METRIC)
    ax1.legend(fontsize=14)
    _format_time_axis(ax1, data_small)
 
    # -- Panel 2: Total RMSE - Large gaps (Radials vs Totals) -----------------
    ax2 = axes[2]
    ax2.plot(data_large['fecha'], data_large['total_rmse_rad'],
             label='DIVAnd Radials', linewidth=2.5,marker='o',markersize=4, color=COLOR_RAD)
    ax2.plot(data_large['fecha'], data_large['total_rmse_tot'],
             label='DIVAnd Totals',  linewidth=2.5, marker='o',markersize=4,color=COLOR_TOT)
    ax2.set_ylabel('Total RMSE (m/s)', fontsize=16, labelpad=12)
    ax2.set_title('Total RMSE - Large gaps (>=4)', fontsize=18, fontweight='bold')
    ax2.set_ylim(YLIM_METRIC)
    ax2.legend(fontsize=14)
    _format_time_axis(ax2, data_large)
 
    plt.tight_layout()
    _save(fig, archivo_salida)
 
 
# --- Horizontal panel: RMSE by gap size ---

def plot_panel_rmse_horizontal(data_small, data_large, archivo_salida):
    """
    Panel horizontal 1×2: Total RMSE small gaps | Total RMSE large gaps.
    - Solo líneas (sin markers)
    - Sin títulos (ni suptitle ni ax.set_title)
    - Una sola leyenda compartida, centrada entre los dos paneles
    - Gridlines solo en los días (medianoche); sin ticks menores
    - Ticks mayores cada día, etiqueta solo con el número de día
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(34, 14), sharex=False)

    def _format_day_axis(ax, data):
        """Eje X: ticks y grid solo en días, etiqueta = número de día."""
        if data.empty:
            return

        fecha_min = data['date'].min()
        fecha_max = data['date'].max()
        fecha_min_floor = datetime(fecha_min.year, fecha_min.month, fecha_min.day)
        fecha_max_ceil  = datetime(fecha_max.year, fecha_max.month, fecha_max.day) + timedelta(days=1)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))  # tick menor a las 12h
        ax.set_xlim(fecha_min_floor, fecha_max_ceil)

        ax.grid(True,  which='major', alpha=0.3)           # grid solo en días
        ax.grid(False, which='minor')

        ax.tick_params(axis='x', which='major', length=10, width=1.5,
                       color='black', direction='out', labelsize=32,
                       bottom=True, pad=6)
        ax.tick_params(axis='x', which='minor', length=6, width=1.0,
                       color='black', direction='out', bottom=True,
                       labelbottom=False)  # tick visible, sin etiqueta
        ax.tick_params(axis='y', which='both',  labelsize=32, length=8,
                       width=1.2, color='black', direction='out')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45,
                 ha='right', rotation_mode='anchor')

        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.2)

    # -- Panel izquierdo: Small gaps ------------------------------------------
    l1, = ax1.plot(data_small['fecha'], data_small['total_rmse_rad'],
                   linewidth=2.5, color=COLOR_RAD, label='DIVAnd Radials')
    l2, = ax1.plot(data_small['fecha'], data_small['total_rmse_tot'],
                   linewidth=2.5, color=COLOR_TOT, label='DIVAnd Totals')
    ax1.set_ylabel('Total RMSE (m/s)', fontsize=32, labelpad=12)
    ax1.set_ylim(YLIM_METRIC)
    _format_day_axis(ax1, data_small)

    # -- Panel derecho: Large gaps --------------------------------------------
    ax2.plot(data_large['fecha'], data_large['total_rmse_rad'],
             linewidth=2.5, color=COLOR_RAD)
    ax2.plot(data_large['fecha'], data_large['total_rmse_tot'],
             linewidth=2.5, color=COLOR_TOT)
    #ax2.set_ylabel('Total RMSE (m/s)', fontsize=24, labelpad=12)
    ax2.set_ylim(YLIM_METRIC)
    _format_day_axis(ax2, data_large)

    # -- Leyenda única centrada abajo entre los dos paneles -------------------
    fig.legend(handles=[l1, l2], fontsize=32,
               loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.08),
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, archivo_salida)


# --- Comparative plots: Small vs Large gaps ---
 
def plot_comparison(data_small, data_large,
                          col_rad, col_tot,
                          ylabel, titulo, archivo_salida,
                          apply_metric_ylim=True):
    """
    1×2 panel: left = Radials (small vs large),
               right   = Totals  (small vs large).
    apply_metric_ylim: if True apply YLIM_METRIC (0.1-1.0 m/s).
    """
    fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(22, 7), sharex=False)
    fig.suptitle(titulo, fontsize=22, fontweight='bold', y=1.01)
 
    for ax, col, method_label in [(ax_r, col_rad, 'Radials'),
                                   (ax_t, col_tot, 'Totals')]:
        ax.plot(data_small['fecha'], data_small[col],
                label='Small gaps (<4)',  linewidth=2.5, color=COLOR_SMALL)
        ax.plot(data_large['fecha'], data_large[col],
                label='Large gaps (>=4)', linewidth=2.5, color=COLOR_LARGE)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=12)
        ax.set_title(method_label, fontsize=18, fontweight='bold')
        if apply_metric_ylim:
            ax.set_ylim(YLIM_METRIC)
        ax.legend(fontsize=14)
        _format_time_axis(ax, data_small)
 
    plt.tight_layout()
    _save(fig, archivo_salida)
 
 
def plot_panel_comparison_rmse_mae(data_small, data_large, archivo_salida):
    """
    3×4 panel: row = RMSE / MAE / STD,
               columns = U-small / U-large / V-small / V-large.
    Each sub-plot compares DIVAnd Radials with DIVAnd Totals
	for that specific gap size.
    """
    fig, axes = plt.subplots(3, 4, figsize=(28, 16), sharex=False)
    fig.suptitle('RMSE, MAE & STD — Radials vs Totals  (Small | Large gaps)',
                 fontsize=22, fontweight='bold', y=1.01)
 
    # layout: (ax, data, col_rad, col_tot, ylabel)
    layout = [
        # row 0: RMSE
        (axes[0, 0], data_small, 'rmse_u_rad',     'rmse_u_tot',     'RMSE U  Small (m/s)'),
        (axes[0, 1], data_large, 'rmse_u_rad',     'rmse_u_tot',     'RMSE U  Large (m/s)'),
        (axes[0, 2], data_small, 'rmse_v_rad',     'rmse_v_tot',     'RMSE V  Small (m/s)'),
        (axes[0, 3], data_large, 'rmse_v_rad',     'rmse_v_tot',     'RMSE V  Large (m/s)'),
        # row 1: MAE
        (axes[1, 0], data_small, 'mae_u_rad',      'mae_u_tot',      'MAE U   Small (m/s)'),
        (axes[1, 1], data_large, 'mae_u_rad',      'mae_u_tot',      'MAE U   Large (m/s)'),
        (axes[1, 2], data_small, 'mae_v_rad',      'mae_v_tot',      'MAE V   Small (m/s)'),
        (axes[1, 3], data_large, 'mae_v_rad',      'mae_v_tot',      'MAE V   Large (m/s)'),
        # row 2: STD
        (axes[2, 0], data_small, 'std_u_rad',      'std_u_tot',      'STD U   Small (m/s)'),
        (axes[2, 1], data_large, 'std_u_rad',      'std_u_tot',      'STD U   Large (m/s)'),
        (axes[2, 2], data_small, 'std_v_rad',      'std_v_tot',      'STD V   Small (m/s)'),
        (axes[2, 3], data_large, 'std_v_rad',      'std_v_tot',      'STD V   Large (m/s)'),
    ]
 
    for ax, data, col_rad, col_tot, ylabel in layout:
        ax.plot(data['date'], data[col_rad],
                label='Radials', linewidth=2, color=COLOR_RAD)
        ax.plot(data['date'], data[col_tot],
                label='Totals',  linewidth=2, color=COLOR_TOT)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(YLIM_METRIC)
        ax.legend(fontsize=10)
        _format_time_axis(ax, data)
 
    plt.tight_layout()
    _save(fig, archivo_salida)
 
 
# --- Summary statistics ---
 
def print_statistics(data, gap_label):
    """Prints basic summary statistics (mean, median, max) to the console."""
    print("\n" + "=" * 80)
    print(f"DATA STATISTICS  [{gap_label}]")
    print("=" * 80)
    print(f"\nNumber of records:  {len(data)}")
    print(f"Temporal range      : {data['time'].min()} - {data['time'].max()} horas")
    print(f"Dates              : {data['date'].min()} - {data['date'].max()}")
 
    sections = [
        ("N pixels",     'N_rad',          'N_tot'),
        ("AVG U",        'avg_u_rad',      'avg_u_tot'),
        ("AVG V",        'avg_v_rad',      'avg_v_tot'),
        ("AVG MAG",      'avg_mag_rad',    'avg_mag_tot'),
        ("RMSE U",       'rmse_u_rad',     'rmse_u_tot'),
        ("RMSE V",       'rmse_v_rad',     'rmse_v_tot'),
        ("RMSE TOTAL",   'total_rmse_rad', 'total_rmse_tot'),
        ("MAE U",        'mae_u_rad',      'mae_u_tot'),
        ("MAE V",        'mae_v_rad',      'mae_v_tot'),
        ("MAE TOTAL",    'total_mae_rad',  'total_mae_tot'),
        ("STD U",        'std_u_rad',      'std_u_tot'),
        ("STD V",        'std_v_rad',      'std_v_tot'),
        ("STD TOTAL",    'std_tot_rad',    'std_tot_tot'),
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
            s = data[col].dropna()
            print(f"{title:<14} {label:<10} {s.mean():>14.6e} {s.median():>14.6e} {s.max():>14.6e}")
 
    print("\n" + "=" * 80)
 
 
# --- Generate all plots ---
 
def create_all_plots(data_small, data_large, output_dir):
    """Creates all individual and comparative plots and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
 
    def path(nombre):
        return os.path.join(output_dir, nombre)
 
    print("\nGenerating plots — Small gaps...")
    for fn, col, name in [
        (plot_avg_u,     None, 'avg_u'),
        (plot_avg_v,     None, 'avg_v'),
        (plot_avg_mag,   None, 'avg_mag'),
        (plot_rmse_u,    None, 'rmse_u'),
        (plot_rmse_v,    None, 'rmse_v'),
        (plot_total_rmse,None, 'total_rmse'),
        (plot_mae_u,     None, 'mae_u'),
        (plot_mae_v,     None, 'mae_v'),
        (plot_total_mae, None, 'total_mae'),
        (plot_std_u,     None, 'std_u'),
        (plot_std_v,     None, 'std_v'),
        (plot_total_std, None, 'total_std'),
        (plot_N,         None, 'N_pixels'),
    ]:
        fn(data_small, 'Small gaps (<4)', path(f'small_{name}.png'))
 
    plot_panel_rmse_mae(data_small, 'Small gaps (<4)', path('small_panel_rmse_mae.png'))
 
    print("\nGenerating plots — Large gaps...")
    for fn, name in [
        (plot_avg_u,      'avg_u'),
        (plot_avg_v,      'avg_v'),
        (plot_avg_mag,    'avg_mag'),
        (plot_rmse_u,     'rmse_u'),
        (plot_rmse_v,     'rmse_v'),
        (plot_total_rmse, 'total_rmse'),
        (plot_mae_u,      'mae_u'),
        (plot_mae_v,      'mae_v'),
        (plot_total_mae,  'total_mae'),
        (plot_std_u,      'std_u'),
        (plot_std_v,      'std_v'),
        (plot_total_std,  'total_std'),
        (plot_N,          'N_pixels'),
    ]:
        fn(data_large, 'Large gaps (>=4)', path(f'large_{name}.png'))
 
    plot_panel_rmse_mae(data_large, 'Large gaps (>=4)', path('large_panel_rmse_mae.png'))
 
    print("\nGenerating comparative plots Small vs Large...")
 
    comp_vars = [
        ('avg_u_rad',      'avg_u_tot',      'Average bias U (m/s)',        'Avg bias U — Small vs Large',          'comp_avg_u.png',       False),
        ('avg_v_rad',      'avg_v_tot',      'Average bias V (m/s)',        'Avg bias V — Small vs Large',          'comp_avg_v.png',       False),
        ('avg_mag_rad',    'avg_mag_tot',    'Average bias |vel| (m/s)',    'Avg bias magnitude — Small vs Large',  'comp_avg_mag.png',     False),
        ('rmse_u_rad',     'rmse_u_tot',     'RMSE U (m/s)',                'RMSE U — Small vs Large',              'comp_rmse_u.png',      True),
        ('rmse_v_rad',     'rmse_v_tot',     'RMSE V (m/s)',                'RMSE V — Small vs Large',              'comp_rmse_v.png',      True),
        ('total_rmse_rad', 'total_rmse_tot', 'Total RMSE (m/s)',            'Total RMSE — Small vs Large',          'comp_total_rmse.png',  True),
        ('mae_u_rad',      'mae_u_tot',      'MAE U (m/s)',                 'MAE U — Small vs Large',               'comp_mae_u.png',       True),
        ('mae_v_rad',      'mae_v_tot',      'MAE V (m/s)',                 'MAE V — Small vs Large',               'comp_mae_v.png',       True),
        ('total_mae_rad',  'total_mae_tot',  'Total MAE (m/s)',             'Total MAE — Small vs Large',           'comp_total_mae.png',   True),
        ('std_u_rad',      'std_u_tot',      'STD U (m/s)',                 'STD U — Small vs Large',               'comp_std_u.png',       True),
        ('std_v_rad',      'std_v_tot',      'STD V (m/s)',                 'STD V — Small vs Large',               'comp_std_v.png',       True),
        ('std_tot_rad',    'std_tot_tot',    'Total STD (m/s)',             'Total STD — Small vs Large',           'comp_total_std.png',   True),
    ]
 
    for col_rad, col_tot, ylabel, titulo, fname, use_ylim in comp_vars:
        plot_comparison(data_small, data_large,
                             col_rad, col_tot,
                             ylabel, titulo, path(fname),
                             apply_metric_ylim=use_ylim)
 
    plot_panel_comparison_rmse_mae(data_small, data_large,
                                        path('comp_panel_rmse_mae.png'))
 
    plot_panel_N_and_rmse(data_small, data_large,
                             path('comp_panel_N_y_rmse.png'))

    plot_panel_rmse_horizontal(data_small, data_large,
                                   path('comp_panel_rmse_horizontal.png'))
 
    print("\nPlots generated successfully.")
 
 
# --- Main entry point ---
 
if __name__ == "__main__":
 
        # --- Configuration ---
    base_dir   = "../../data/january_2026/gaps_data"
    file_small = os.path.join(base_dir, "gaps_stats_small_time_series.txt")
    file_large = os.path.join(base_dir, "gaps_stats_large_time_series.txt")
 
    date_ini   = '2026-01-14 00:00:00'
    output_dir = '../../figures/january_2026/gaps_by_size/'
 
    print("=" * 80)
    print("METRICS ANALYSIS BY GAP SIZE — RADIALS vs TOTALS")
    print("=" * 80)
 
    # Read data
    print(f"\nReading small gap data: {file_small}")
    data_small = read_data(file_small)
    data_small = creating_dates(data_small, date_ini)
 
    print(f"Reading large gap data: {file_large}")
    data_large = read_data(file_large)
    data_large = creating_dates(data_large, date_ini)
 
    # Estadísticas en pantalla
    print_statistics(data_small, 'Small gaps (<4)')
    print_statistics(data_large, 'Large gaps (>=4)')
 
    # Generate plots
    create_all_plots(data_small, data_large, output_dir=output_dir)
 
    print("\nProcess completed successfully.")
    print("=" * 80)
