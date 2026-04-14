import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import os

# -----------------------------------------------------------------------
# Script: plot_stats_10_days_icatmar_grid.py
# Purpose: Loads a time series of domain-averaged physical statistics
#          (divergence, vorticity, kinetic energy, and their correlations
#          with the Copernicus MEDSEA model) for three reconstruction
#          methods (DIVAnd-radials, DIVAnd-totals, Least Squares) and
#          generates individual time series figures for each variable.
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

# ------------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------------

# Load the statistics text file (columns: time, divergence, vorticity, KE, correlations)
data = np.loadtxt('../../data/january_2026/stats_data/all_stats_phys_grid_icatmar.txt')

# Reference start date for the January 2026 time series
start_date = datetime(2026, 1, 14, 0, 0, 0)

# Build the DataFrame with descriptive column names
df = pd.DataFrame({
    'date':         [start_date + timedelta(hours=h) for h in data[:, 0]],
    'time_hours':   data[:, 0],
    'div_model':    data[:, 1],    # Divergence - Copernicus MEDSEA
    'div_rad':      data[:, 2],    # Divergence - DIVAnd radials
    'div_tot':      data[:, 3],    # Divergence - DIVAnd totals
    'div_LS':       data[:, 4],    # Divergence - Least Squares
    'vort_model':   data[:, 5],    # Vorticity - Copernicus MEDSEA
    'vort_rad':     data[:, 6],    # Vorticity - DIVAnd radials
    'vort_tot':     data[:, 7],    # Vorticity - DIVAnd totals
    'vort_LS':      data[:, 8],    # Vorticity - Least Squares
    'ke_model':     data[:, 9],    # Kinetic energy - Copernicus MEDSEA
    'ke_rad':       data[:, 10],   # Kinetic energy - DIVAnd radials
    'ke_tot':       data[:, 11],   # Kinetic energy - DIVAnd totals
    'ke_LS':        data[:, 12],   # Kinetic energy - Least Squares
    'div_corr_rad': data[:, 13],   # Divergence correlation - DIVAnd radials vs MEDSEA
    'div_corr_tot': data[:, 14],   # Divergence correlation - DIVAnd totals vs MEDSEA
    'div_corr_LS':  data[:, 15],   # Divergence correlation - LS vs MEDSEA
    'vort_corr_rad':data[:, 16],   # Vorticity correlation - DIVAnd radials vs MEDSEA
    'vort_corr_tot':data[:, 17],   # Vorticity correlation - DIVAnd totals vs MEDSEA
    'vort_corr_LS': data[:, 18],   # Vorticity correlation - LS vs MEDSEA
    'ke_corr_rad':  data[:, 19],   # KE correlation - DIVAnd radials vs MEDSEA
    'ke_corr_tot':  data[:, 20],   # KE correlation - DIVAnd totals vs MEDSEA
    'ke_corr_LS':   data[:, 21],   # KE correlation - LS vs MEDSEA
})

print(df.head())
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"Number of time steps: {len(df)}")

# ------------------------------------------------------------------------------------
# COLOUR SCHEME (consistent across all plots)
# ------------------------------------------------------------------------------------

colors = {
    'MEDSEA':         'black',
    'Radials':        'red',
    'Totals':         'blue',
    'LS':             'orange',
    'DIVAnd-Rad':     'red',
    'DIVAnd-Tot':     'blue',
    'DIVAnd-Radials': 'red',
    'DIVAnd-Totals':  'blue'
}

date_formatter = mdates.DateFormatter('%d')

# ------------------------------------------------------------------------------------
# INDIVIDUAL PLOT FUNCTION
# ------------------------------------------------------------------------------------

def plot_individual(df, data_dict, ylabel, title, filename, ylim=None, show_legend=False):
    """
    Creates and saves a single time series figure for a given set of variables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a 'date' column and the data columns to plot
    data_dict : dict
        Mapping of legend labels to DataFrame column names,
        e.g. {'MEDSEA': 'div_model', 'LS': 'div_LS', ...}
    ylabel : str
        Y-axis label
    title : str
        Title text displayed in a box in the lower-right corner of the plot
    filename : str
        Output file path (with extension)
    ylim : tuple, optional
        Y-axis limits (min, max)
    show_legend : bool, optional
        Whether to display the legend (default: False)
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot each method's time series
    for label, col_name in data_dict.items():
        # MEDSEA reference line is thinner than the reconstruction method lines
        lw = 2 if label == 'MEDSEA' else 4
        ax.plot(df['date'], df[col_name], label=label, linewidth=lw,
                color=colors.get(label, 'gray'))

    # --- X-axis time formatting ---
    date_min       = df['date'].min()
    date_max       = df['date'].max()
    date_min_floor = datetime(date_min.year, date_min.month, date_min.day)
    date_max_ceil  = datetime(date_max.year, date_max.month, date_max.day) + timedelta(days=1)

    # Major ticks every day at midnight; minor ticks every 12 hours
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))

    ax.set_xlim(date_min_floor, date_max_ceil)

    # Grid only on major (daily) ticks
    ax.grid(True, which='major', alpha=0.3)

    # Tick appearance
    ax.tick_params(axis='x', which='minor', length=12, width=1,   color='black', direction='out')
    ax.tick_params(axis='x', which='major', length=6,  width=1.5, color='black', direction='out')
    ax.tick_params(axis='y', which='both',  labelsize=28, length=8, width=1.2, color='black', direction='out')

    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='x', which='major', bottom=True)

    ax.set_ylabel(ylabel, fontsize=28, labelpad=14)

    if show_legend:
        ax.legend(fontsize=28, loc="lower left")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right',
             rotation_mode='anchor', fontsize=28)

    # Diagnostic tick counts
    major_ticks = ax.xaxis.get_majorticklocs()
    minor_ticks = ax.xaxis.get_minorticklocs()
    print(f"Major ticks (days): {len(major_ticks)} ticks")
    print(f"Minor ticks (12h):  {len(minor_ticks)} ticks")

    # Title placed as a text box in the lower-right corner
    ax.text(0.98, 0.1, title, transform=ax.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                      edgecolor='black', linewidth=3))

    if ylim:
        ax.set_ylim(ylim)

    # Horizontal reference line at zero
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax.xaxis.set_major_formatter(date_formatter)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close()

# ------------------------------------------------------------------------------------
# GENERATE ALL FIGURES
# ------------------------------------------------------------------------------------

# Create the output directory if it does not exist
output_dir = '../../figures/january_2026/physics'
os.makedirs(output_dir, exist_ok=True)

# --- Divergence time series ---
plot_individual(df,
    {'MEDSEA': 'div_model', 'LS': 'div_LS', 'Totals': 'div_tot', 'Radials': 'div_rad'},
    r'Divergence (s$^{-1}$)', 'Average divergence time series',
    os.path.join(output_dir, 'divergence.png'))

# --- Vorticity time series ---
plot_individual(df,
    {'MEDSEA': 'vort_model', 'LS': 'vort_LS', 'Totals': 'vort_tot', 'Radials': 'vort_rad'},
    r'Vorticity (s$^{-1}$)', 'Average vorticity time series',
    os.path.join(output_dir, 'vorticity.png'))

# --- Kinetic energy time series ---
plot_individual(df,
    {'MEDSEA': 'ke_model', 'LS': 'ke_LS', 'Totals': 'ke_tot', 'Radials': 'ke_rad'},
    r'Kinetic energy (m$^2$/s$^2$)', 'Average kinetic energy time series',
    os.path.join(output_dir, 'ke.png'))

# --- Divergence correlation with MEDSEA ---
plot_individual(df,
    {'LS': 'div_corr_LS', 'Totals': 'div_corr_tot', 'Radials': 'div_corr_rad'},
    'Correlation coefficient', 'Divergence',
    os.path.join(output_dir, 'divergence_correlations.png'), ylim=[0.4, 1])

# --- Vorticity correlation with MEDSEA ---
plot_individual(df,
    {'LS': 'vort_corr_LS', 'Totals': 'vort_corr_tot', 'Radials': 'vort_corr_rad'},
    'Correlation coefficient', 'Vorticity',
    os.path.join(output_dir, 'vorticity_correlations.png'), ylim=[0.4, 1])

# --- Kinetic energy correlation with MEDSEA (with legend) ---
plot_individual(df,
    {'LS': 'ke_corr_LS', 'Totals': 'ke_corr_tot', 'Radials': 'ke_corr_rad'},
    'Correlation coefficient', 'Kinetic energy',
    os.path.join(output_dir, 'ke_correlations.png'), ylim=[0.4, 1], show_legend=True)

print("\nAll figures generated successfully!")
