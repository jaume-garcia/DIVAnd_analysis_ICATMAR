import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
 
# -----------------------------------------------------------------------
# Module: fig_vel.py
# Purpose: Collection of plotting functions for oceanographic velocity
#          fields derived from HF radar and numerical models.
#          All functions save figures to disk in PNG format.
# -----------------------------------------------------------------------
 
 
def vel_quiver(lon_mesh, lat_mesh, u, v, lon_bat, lat_bat, h, pdt, scale, width,
               title_name, fig_name, lon_min, lon_max, lat_min, lat_max, mask=[]):
    """
    Generates a quiver (arrow) plot of a 2D velocity field overlaid on bathymetry.
 
    Parameters:
    -----------
    lon_mesh, lat_mesh : ndarray
        2D longitude and latitude grids for the velocity field
    u, v : ndarray
        East and north velocity components on the grid
    lon_bat, lat_bat : ndarray
        Longitude and latitude grids for the bathymetry
    h : ndarray
        Bathymetric depth values
    pdt : int
        Decimation factor for the quiver plot (every pdt-th point is shown)
    scale : float
        Scale factor for arrow lengths
    width : float
        Width of the arrows
    title_name : str
        Plot title
    fig_name : str
        Output filename (without extension)
    lon_min, lon_max : float
        Longitude axis limits
    lat_min, lat_max : float
        Latitude axis limits
    mask : ndarray, optional
        Land/sea mask; a contour is drawn at the boundary if provided
    """
    fig, axes = plt.subplots(figsize=(9, 9), subplot_kw={'projection': ccrs.PlateCarree()})
 
    # Plot bathymetry as a filled colour map
    plt.pcolor(lon_bat, lat_bat, h, cmap="Blues_r")
 
    # Plot velocity arrows (decimated by pdt)
    Q = axes.quiver(lon_mesh[::pdt, ::pdt], lat_mesh[::pdt, ::pdt],
                    u[::pdt, ::pdt], v[::pdt, ::pdt],
                    transform=ccrs.PlateCarree(), scale=scale, width=width)
 
    # Optionally draw the land/sea mask boundary
    if len(mask) != 0:
        axes.contour(lon_mesh[:, :], lat_mesh[:, :], mask, 1,
                     linewidths=0.5, colors='black')
 
    axes.set_xlim(lon_min, lon_max)
    axes.set_ylim(lat_min, lat_max)
 
    # Reference arrow key (0.5 m/s)
    axes.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
    axes.coastlines()
    axes.set_title(title_name)
    plt.xlabel("Longitude (°N)")
    plt.ylabel("Latitude (°E)")
    plt.tight_layout()
    plt.savefig(fig_name + ".png")
 
    return
 
 
def vel_quiver_eta(lon_mesh, lat_mesh, zeta_val, u, v, lon_bat, lat_bat, h,
                   zeta_min, zeta_max, pdt, scale, width, title_name, fig_name, mask=[]):
    """
    Generates a quiver plot with sea surface height (SSH / eta) contours
    overlaid on bathymetry.
 
    Parameters:
    -----------
    lon_mesh, lat_mesh : ndarray
        2D longitude and latitude grids
    zeta_val : ndarray
        Sea surface height (SSH) field
    u, v : ndarray
        East and north velocity components
    lon_bat, lat_bat : ndarray
        Bathymetry grid coordinates
    h : ndarray
        Bathymetric depth values
    zeta_min, zeta_max : float
        SSH colour/contour range limits
    pdt : int
        Arrow decimation factor
    scale, width : float
        Quiver scale and arrow width
    title_name : str
        Plot title
    fig_name : str
        Output filename (without extension)
    mask : ndarray, optional
        Land/sea mask for boundary contour
    """
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
 
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
 
    # Bathymetry background
    plt.pcolor(lon_bat, lat_bat, h, cmap="Blues_r")
 
    # Optionally draw the land/sea mask boundary
    if len(mask) != 0:
        ax.contour(lon_mesh[:, :], lat_mesh[:, :], mask, 1, linewidths=0.5, colors='black')
 
    # SSH contours
    ax.contour(lon_mesh, lat_mesh, zeta_val[:, :], levels=9,
               colors="xkcd:green", vmin=zeta_min, vmax=zeta_max)
 
    # Velocity quiver
    Q = ax.quiver(lon_mesh[::pdt, ::pdt], lat_mesh[::pdt, ::pdt],
                  u[::pdt, ::pdt], v[::pdt, ::pdt],
                  transform=ccrs.PlateCarree(), scale=scale, width=width)
 
    # Redraw mask boundary on top of quiver
    ax.contour(lon_mesh, lat_mesh, mask, 1, linewidths=0.5, colors='black')
    ax.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
 
    plt.title(title_name)
    plt.tight_layout()
    plt.savefig(fig_name + ".png")
    plt.show()
 
    return
 
 
def quiver_vel_rad_obs(lon_vel, lat_vel, u_vel, v_vel,
                       lon_socib, lat_socib, u_socib, v_socib,
                       lon_radar, lat_radar, u_radar, v_radar,
                       lon_bat, lat_bat, h,
                       pdt, pdt1, scale, width,
                       x_min, x_max, y_min, y_max,
                       title_name, fig_name, mask=[]):
    """
    Plots a modelled (or geostrophic) velocity field together with two sets
    of radar observations (HF radar and SOCIB glider/drifter data).
 
    Parameters:
    -----------
    lon_vel, lat_vel : ndarray
        Grid coordinates for the modelled velocity field
    u_vel, v_vel : ndarray
        Modelled east and north velocity components
    lon_socib, lat_socib : ndarray
        Grid coordinates for SOCIB observations
    u_socib, v_socib : ndarray
        SOCIB east and north velocity components
    lon_radar, lat_radar : ndarray
        Grid coordinates for HF radar observations
    u_radar, v_radar : ndarray
        HF radar east and north velocity components
    lon_bat, lat_bat : ndarray
        Bathymetry grid coordinates
    h : ndarray
        Bathymetric depth values
    pdt : int
        Decimation factor for the model field
    pdt1 : int
        Decimation factor for the observation fields
    scale, width : float
        Quiver scale and arrow width
    x_min, x_max, y_min, y_max : float
        Axis limits
    title_name : str
        Plot title
    fig_name : str
        Output filename (without extension)
    mask : ndarray, optional
        Land/sea mask for boundary contour
    """
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
 
    ax.coastlines()
 
    # Bathymetry with colour bar
    pcm = plt.pcolor(lon_bat, lat_bat, h, cmap="Blues_r")
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label="Depth (km)",
                        fraction=0.03, pad=0.03)
 
    # Land/sea mask boundary
    if len(mask) != 0:
        ax.contour(lon_vel[:, :], lat_vel[:, :], mask, 1, linewidths=0.5, colors='black')
 
    # Model velocity field (black arrows)
    Q = ax.quiver(lon_vel[::pdt, ::pdt], lat_vel[::pdt, ::pdt],
                  u_vel[::pdt, ::pdt], v_vel[::pdt, ::pdt],
                  transform=ccrs.PlateCarree(), scale=scale, width=width, minlength=0)
 
    # HF radar observations (red, semi-transparent)
    Q1 = ax.quiver(lon_radar[::pdt1, ::pdt1], lat_radar[::pdt1, ::pdt1],
                   u_radar[::pdt1, ::pdt1], v_radar[::pdt1, ::pdt1],
                   transform=ccrs.PlateCarree(), scale=scale,
                   width=width, minlength=0, color="red", alpha=0.5)
 
    # SOCIB observations (red, semi-transparent)
    Q2 = ax.quiver(lon_socib[::pdt1, ::pdt1], lat_socib[::pdt1, ::pdt1],
                   u_socib[::pdt1, ::pdt1], v_socib[::pdt1, ::pdt1],
                   transform=ccrs.PlateCarree(), scale=scale,
                   width=width, minlength=0, color="red", alpha=0.5)
 
    ax.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
 
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
 
    plt.title(title_name)
    plt.tight_layout()
    plt.savefig(fig_name + ".png")
 
    return
 
 
def quiver_vel_obs(lon_vel, lat_vel, u_vel, v_vel,
                   lon_socib, lat_socib, u_socib, v_socib,
                   lon_radar, lat_radar, u_radar, v_radar,
                   lon_buoys, lat_buoys, u_buoys, v_buoys,
                   lon_bat, lat_bat, h,
                   pdt, pdt1, scale, width,
                   x_min, x_max, y_min, y_max,
                   title_name, fig_name, mask=[]):
    """
    Plots a modelled (or geostrophic) velocity field together with three
    observation sources: HF radar, SOCIB, and drifting buoys.
 
    Parameters:
    -----------
    lon_vel, lat_vel : ndarray
        Grid coordinates for the modelled velocity field
    u_vel, v_vel : ndarray
        Modelled east and north velocity components
    lon_socib, lat_socib : ndarray
        Grid coordinates for SOCIB observations
    u_socib, v_socib : ndarray
        SOCIB velocity components
    lon_radar, lat_radar : ndarray
        Grid coordinates for HF radar observations
    u_radar, v_radar : ndarray
        HF radar velocity components
    lon_buoys, lat_buoys : array-like
        Positions of the drifting buoys
    u_buoys, v_buoys : array-like
        Buoy velocity components
    lon_bat, lat_bat : ndarray
        Bathymetry grid coordinates
    h : ndarray
        Bathymetric depth values
    pdt : int
        Decimation factor for the model field
    pdt1 : int
        Decimation factor for the radar and SOCIB fields
    scale, width : float
        Quiver scale and arrow width
    x_min, x_max, y_min, y_max : float
        Axis limits
    title_name : str
        Plot title
    fig_name : str
        Output filename (without extension)
    mask : ndarray, optional
        Land/sea mask for boundary contour
    """
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
 
    ax.coastlines()
 
    # Bathymetry with colour bar
    pcm = plt.pcolor(lon_bat, lat_bat, h, cmap="Blues_r")
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label="Depth (km)",
                        fraction=0.03, pad=0.03)
 
    # Land/sea mask boundary
    if len(mask) != 0:
        ax.contour(lon_vel[:, :], lat_vel[:, :], mask, 1, linewidths=0.5, colors='black')
 
    # Model velocity field (black, lowest z-order)
    Q = ax.quiver(lon_vel[::pdt, ::pdt], lat_vel[::pdt, ::pdt],
                  u_vel[::pdt, ::pdt], v_vel[::pdt, ::pdt],
                  transform=ccrs.PlateCarree(), scale=scale, width=width, minlength=0, zorder=1)
 
    # HF radar observations (red, semi-transparent)
    Q1 = ax.quiver(lon_radar[::pdt1, ::pdt1], lat_radar[::pdt1, ::pdt1],
                   u_radar[::pdt1, ::pdt1], v_radar[::pdt1, ::pdt1],
                   transform=ccrs.PlateCarree(), scale=scale,
                   width=width, minlength=0, color="red", alpha=0.5, zorder=2)
 
    # SOCIB observations (red, semi-transparent)
    Q2 = ax.quiver(lon_socib[::pdt1, ::pdt1], lat_socib[::pdt1, ::pdt1],
                   u_socib[::pdt1, ::pdt1], v_socib[::pdt1, ::pdt1],
                   transform=ccrs.PlateCarree(), scale=scale,
                   width=width, minlength=0, color="red", alpha=0.5, zorder=3)
 
    # Drifting buoys (red, fully opaque, highest z-order)
    Q3 = ax.quiver(lon_buoys, lat_buoys, u_buoys, v_buoys,
                   transform=ccrs.PlateCarree(), scale=scale,
                   width=width, minlength=0, color="red", alpha=1.0, zorder=4)
 
    ax.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
 
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
 
    plt.title(title_name)
    plt.tight_layout()
    plt.savefig(fig_name + ".png")
 
    return
 
 
def two_plots_geo_model(lon_model, lat_model, u_model, v_model,
                        lon_geo, lat_geo, u_geo, v_geo,
                        lon_bat, lat_bat, h,
                        pdt, scale, width,
                        x_min, x_max, y_min, y_max,
                        title_name1, title_name2, fig_name, mask=[]):
    """
    Creates a side-by-side figure comparing the numerical model velocity field
    (left panel) with the geostrophic velocity field (right panel).
 
    Parameters:
    -----------
    lon_model, lat_model : ndarray
        Grid coordinates for the model velocity field
    u_model, v_model : ndarray
        Model east and north velocity components
    lon_geo, lat_geo : ndarray
        Grid coordinates for the geostrophic velocity field
    u_geo, v_geo : ndarray
        Geostrophic east and north velocity components
    lon_bat, lat_bat : ndarray
        Bathymetry grid coordinates
    h : ndarray
        Bathymetric depth values
    pdt : int
        Arrow decimation factor (applied to both panels)
    scale, width : float
        Quiver scale and arrow width
    x_min, x_max, y_min, y_max : float
        Axis limits (applied to both panels)
    title_name1 : str
        Title for the model (left) panel
    title_name2 : str
        Title for the geostrophic (right) panel
    fig_name : str
        Output filename (with extension)
    mask : ndarray, optional
        Land/sea mask; boundary contour drawn on the right panel if provided
    """
    fig = plt.figure(figsize=(12, 6))
 
    # --- Left panel: numerical model ---
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
 
    # Bathymetry
    pcm = ax0.pcolormesh(lon_bat, lat_bat, h, cmap="Blues_r")
 
    ax0.coastlines()
 
    Q = ax0.quiver(lon_model[::pdt, ::pdt], lat_model[::pdt, ::pdt],
                   u_model[::pdt, ::pdt], v_model[::pdt, ::pdt],
                   transform=ccrs.PlateCarree(), scale=scale, width=width, minlength=0)
 
    ax0.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
 
    ax0.set_title(title_name1)
    ax0.set_xlabel("Longitude (°)", fontsize=10)
    ax0.set_ylabel("Latitude (°)", fontsize=10)
    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)
 
    plt.tight_layout()
 
    # --- Right panel: geostrophic velocity ---
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
 
    # Bathymetry
    pcm1 = ax1.pcolormesh(lon_bat, lat_bat, h, cmap="Blues_r")
 
    ax1.coastlines()
 
    # Optionally draw the land/sea mask boundary
    if len(mask) != 0:
        ax1.contour(lon_geo[:, :], lat_geo[:, :], mask, 1, linewidths=0.5, colors='black')
 
    Q = ax1.quiver(lon_geo[::pdt, ::pdt], lat_geo[::pdt, ::pdt],
                   u_geo[::pdt, ::pdt], v_geo[::pdt, ::pdt],
                   transform=ccrs.PlateCarree(), scale=scale, width=width, minlength=0)
 
    ax1.quiverkey(Q, 0.20, 0.92, 0.5, "0.5 m/s", labelpos='E')
 
    ax1.set_title(title_name2)
    ax1.set_xlabel("Longitude (°)", fontsize=10)
    ax1.set_ylabel("Latitude (°)", fontsize=10)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
 
    plt.tight_layout()
    plt.savefig(fig_name)
 
    return
 
 
def plot_rms(dates, date_format, u_rms, v_rms, total_rms, x_label_name, title_name, fig_name):
    """
    Generates a time series plot of the Root Mean Square Error (RMSE) for
    the u component, v component, and total velocity.
 
    Parameters:
    -----------
    dates : array-like
        Array of datetime objects for the x-axis
    date_format : matplotlib DateFormatter
        Formatter for the date labels on the x-axis
    u_rms : array-like
        RMSE time series for the east (u) component
    v_rms : array-like
        RMSE time series for the north (v) component
    total_rms : array-like
        Total RMSE time series (combined u and v)
    x_label_name : str
        Label for the x-axis
    title_name : str
        Plot title
    fig_name : str
        Output filename (with extension)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, u_rms, 'ro-', label='MSE U-Component')
    plt.plot(dates, v_rms, 'bo-', label='MSE V-Component')
    plt.plot(dates, total_rms, 'go-', label='RMS Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(x_label_name)
    plt.ylabel(r'RMS = $\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$')
    plt.title(title_name)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.grid(True)
    plt.savefig(fig_name, dpi=300)
    plt.close()
 
    return
 
 
def plot_stat(dates, date_format, u_std, v_std, x_label_name, y_label_name, title_name, fig_name):
    """
    Generates a time series plot of the standard deviation for the
    u and v velocity components.
 
    Parameters:
    -----------
    dates : array-like
        Array of datetime objects for the x-axis
    date_format : matplotlib DateFormatter
        Formatter for the date labels
    u_std : array-like
        Standard deviation time series for the east (u) component
    v_std : array-like
        Standard deviation time series for the north (v) component
    x_label_name : str
        Label for the x-axis
    y_label_name : str
        Label for the y-axis
    title_name : str
        Plot title
    fig_name : str
        Output filename (with extension)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, u_std, 'ro-', label='U-component')
    plt.plot(dates, v_std, 'bo-', label='V-Component')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.title(title_name)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.grid(True)
    plt.savefig(fig_name, dpi=300)
    plt.close()
 
    return
 
 
def plot_mb(dates, date_format, u_mb, v_mb, total_mb, x_label_name, title_name, fig_name):
    """
    Generates a time series plot of the Mean Bias (MB) for the u component,
    v component, and total velocity.
 
    Parameters:
    -----------
    dates : array-like
        Array of datetime objects for the x-axis
    date_format : matplotlib DateFormatter
        Formatter for the date labels
    u_mb : array-like
        Mean bias time series for the east (u) component
    v_mb : array-like
        Mean bias time series for the north (v) component
    total_mb : array-like
        Total mean bias time series
    x_label_name : str
        Label for the x-axis
    title_name : str
        Plot title
    fig_name : str
        Output filename (with extension)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, u_mb, 'ro-', label='MB U-component')
    plt.plot(dates, v_mb, 'bo-', label='MB V-component')
    plt.plot(dates, total_mb, 'go-', label='MB Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(x_label_name)
    plt.ylabel(r'MB = $\frac{1}{N}\sum_{i=1}^{N} (y_o - \hat{y}_o)^2 - (y_m - \hat{y}_m)^2$')
    plt.title(title_name)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.grid(True)
    plt.savefig(fig_name, dpi=300)
    plt.close()
 
    return
 
 
def plot_coef_correl(vel_model, vel_obs, x_label_name, y_label_name, title_name, fig_name):
    """
    Generates a scatter plot with a linear regression line to assess the
    correlation between modelled and observed velocities (R² coefficient).
 
    Parameters:
    -----------
    vel_model : array-like
        Modelled velocity values
    vel_obs : array-like
        Observed velocity values
    x_label_name : str
        Label for the x-axis (observations)
    y_label_name : str
        Label for the y-axis (model)
    title_name : str
        Plot title
    fig_name : str
        Output filename (with extension)
    """
    vel_min = min(np.min(vel_model), np.min(vel_obs))
    vel_max = max(np.max(vel_model), np.max(vel_obs))
 
    # Compute linear regression between observations and model
    slope_u, intercept_u, r_value_u, p_value_u, std_err_u = stats.linregress(vel_obs, vel_model)
    r_squared_u = r_value_u**2
 
    # Generate points along the regression line for plotting
    u_line = np.linspace(vel_min, vel_max, 100)
    u_reg_line = slope_u * u_line + intercept_u
 
    plt.figure(figsize=(8, 8))
 
    plt.scatter(vel_obs, vel_model, s=1)
    plt.plot(u_line, u_reg_line, 'r-', label=f'R² = {r_squared_u:.4f}')
    plt.xlim((vel_min, vel_max))
    plt.ylim((vel_min, vel_max))
    plt.grid(True)
    plt.ylabel(y_label_name)
    plt.xlabel(x_label_name)
    plt.title(title_name)
    plt.legend()
    plt.savefig(fig_name, dpi=300)
    plt.close()
 
    return











