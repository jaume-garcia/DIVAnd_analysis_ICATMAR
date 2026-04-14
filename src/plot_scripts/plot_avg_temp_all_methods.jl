using PyCall
using PyPlot
using CSV, DataFrames
using DelimitedFiles
using NCDatasets
using NaNMath

# -----------------------------------------------------------------------
# Script: plot_avg_temp_all_methods.jl
# Purpose: Compares three HF radar velocity reconstruction methods
#          (DIVAnd-radials, DIVAnd-totals, and Least Squares) against
#          the Copernicus MEDSEA model (used as ground truth) over a
#          264-step time series.
#
#          For each time step the script:
#            1. Reads the Copernicus, DIVAnd, and LS velocity snapshots.
#            2. Interpolates all fields to a common LS grid.
#            3. Computes velocity magnitude differences and relative errors.
#            4. Accumulates spatial and temporal statistics.
#
#          After the loop, temporal means and standard deviations of the
#          differences are saved as NetCDF and plain-text files.
# -----------------------------------------------------------------------

# --- Julia utility modules ---
include("../../utils/reading_obs.jl")
include("../../utils/divand_process.jl")
include("../../utils/visualization_vel.jl")

# --- Add the Python utility directory to sys.path and import modules ---
push!(PyVector(pyimport("sys")."path"), "../../utils")

hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# Create an instance of the HFRadarProcessor class (used for grid reading)
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# LOCAL UTILITY FUNCTIONS
# ------------------------------------------------------------------------------------

"""
    nanstd(arr, dim) -> Array

Computes the standard deviation along dimension `dim`, ignoring NaN values.
Returns the square root of the variance computed over non-NaN elements only.

# Arguments
- `arr` : Input array (may contain NaN)
- `dim` : Dimension along which to compute the standard deviation
"""
function nanstd(arr, dim)
    # Compute the NaN-ignoring mean along `dim`
    μ = dropdims(
        sum(x -> isnan(x) ? 0.0 : x, arr, dims=dim) ./
        sum(x -> isnan(x) ? 0 : 1,   arr, dims=dim),
        dims=dim
    )

    # Expand μ for broadcasting against arr
    μ_expanded = reshape(
        μ,
        ntuple(i -> i == dim ? 1 : size(μ, i > dim ? i - 1 : i), ndims(arr))
    )

    # Squared deviations from the mean
    squared_diff = (arr .- μ_expanded) .^ 2

    # NaN-ignoring variance
    variance = dropdims(
        sum(x -> isnan(x) ? 0.0 : x, squared_diff, dims=dim) ./
        sum(x -> isnan(x) ? 0 : 1,   squared_diff, dims=dim),
        dims=dim
    )

    return sqrt.(variance)
end


"""
   snapshot_nanstd(arr, dim) -> Array

Alternative NaN-aware standard deviation that additionally strips NaN values
from the result before returning. Useful when the output array itself may
contain NaN (e.g. due to fully-NaN slices).

# Arguments
- `arr` : Input array (may contain NaN)
- `dim` : Dimension along which to compute the standard deviation
"""
function snapshot_nanstd(arr, dim)
    has_nans = any(isnan, arr)

    # NaN-ignoring mean
    μ = dropdims(
        sum(x -> isnan(x) ? 0.0 : x, arr, dims=dim) ./
        sum(x -> isnan(x) ? 0 : 1,   arr, dims=dim),
        dims=dim
    )

    # Expand μ for broadcasting
    μ_expanded = reshape(
        μ,
        ntuple(i -> i == dim ? 1 : size(μ, i > dim ? i - 1 : i), ndims(arr))
    )

    # Squared deviations
    squared_diff = (arr .- μ_expanded) .^ 2

    # NaN-ignoring variance
    variance = dropdims(
        sum(x -> isnan(x) ? 0.0 : x, squared_diff, dims=dim) ./
        sum(x -> isnan(x) ? 0 : 1,   squared_diff, dims=dim),
        dims=dim
    )

    if has_nans
        # Remove NaN entries from the result
        valid_mask     = .!(isnan.(variance))
        variance_clean = variance[valid_mask]
        return sqrt.(variance_clean)
    else
        return sqrt.(variance)
    end
end


"""
    plot_colormesh_triple(xi, yi, diff1, diff2, diff3, mask,
                          title1, title2, title3, outfile_name)

Creates a three-panel vertical colour-mesh figure (one panel per
reconstruction method) using a PlateCarree projection. A single shared
colour bar is placed below the bottom panel.

# Arguments
- `xi, yi`           : 2D longitude and latitude grids
- `diff1, diff2, diff3` : Scalar fields to display (one per method)
- `mask`             : Land/sea mask (values > 0.5 = ocean)
- `title1..3`        : Panel titles (currently unused; kept for API compatibility)
- `outfile_name`     : Output filepath (with extension)
"""
function plot_colormesh_triple(xi, yi, diff1, diff2, diff3, mask,
                                title1, title2, title3, outfile_name)
    fig = figure(figsize=(12, 18))

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    # ── Top panel ────────────────────────────────────────────────────
    ax1 = subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_aspect("equal", adjustable="box")

    pc1 = ax1.pcolormesh(xi, yi, diff1, shading="auto", cmap="viridis",
                          vmin=-0.2, vmax=0.2,
                          transform=ccrs.PlateCarree(), zorder=1)

    ax1.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                 transform=ccrs.PlateCarree(), zorder=1)
    ax1.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                 transform=ccrs.PlateCarree(), zorder=2)

    gl1 = ax1.gridlines(draw_labels=true, dms=true,
                         x_inline=false, y_inline=false,
                         linewidth=0.5, alpha=0.5, linestyle="--")
    gl1.top_labels    = true
    gl1.bottom_labels = false
    gl1.left_labels   = true
    gl1.right_labels  = true

    ax1.tick_params(axis="both", which="major", labelsize=20)
    ax1.set_extent([1.2, 4.26, 40.5, 43.06])

    # ── Middle panel ─────────────────────────────────────────────────
    ax2 = subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax2.set_aspect("equal", adjustable="box")

    pc2 = ax2.pcolormesh(xi, yi, diff2, shading="auto", cmap="viridis",
                          vmin=-0.2, vmax=0.2,
                          transform=ccrs.PlateCarree(), zorder=1)

    ax2.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                 transform=ccrs.PlateCarree(), zorder=1)
    ax2.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                 transform=ccrs.PlateCarree(), zorder=2)

    gl2 = ax2.gridlines(draw_labels=true, dms=true,
                         x_inline=false, y_inline=false,
                         linewidth=0.5, alpha=0.5, linestyle="--")
    gl2.top_labels    = false
    gl2.bottom_labels = false
    gl2.left_labels   = true
    gl2.right_labels  = true

    ax2.tick_params(axis="both", which="major", labelsize=20)
    ax2.set_extent([1.2, 4.26, 40.5, 43.06])

    # ── Bottom panel ─────────────────────────────────────────────────
    ax3 = subplot(3, 1, 3, projection=ccrs.PlateCarree())
    ax3.set_aspect("equal", adjustable="box")

    pc3 = ax3.pcolormesh(xi, yi, diff3, shading="auto", cmap="viridis",
                          vmin=-0.2, vmax=0.2,
                          transform=ccrs.PlateCarree(), zorder=1)

    ax3.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                 transform=ccrs.PlateCarree(), zorder=1)
    ax3.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                 transform=ccrs.PlateCarree(), zorder=2)

    gl3 = ax3.gridlines(draw_labels=true, dms=true,
                         x_inline=false, y_inline=false,
                         linewidth=0.5, alpha=0.5, linestyle="--")
    gl3.top_labels    = false
    gl3.bottom_labels = true
    gl3.left_labels   = true
    gl3.right_labels  = true

    ax3.tick_params(axis="both", which="major", labelsize=20)
    ax3.set_extent([1.2, 4.26, 40.5, 43.06])

    tight_layout()

    # Shared colour bar placed below the bottom panel
    pos    = ax3.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.03, pos.width, 0.015])
    cb = colorbar(pc3, cax=cbar_ax, orientation="horizontal")
    cb.set_label("m/s", fontsize=12)

    savefig(outfile_name)
    close()
end


"""
    plot_colormesh(xi, yi, diff, mask, title_name, outfile_name;
                   show_colorbar=true)

Creates a single-panel colour-mesh map of a scalar velocity difference field
using a diverging RdBu_r colour map (range ±0.2 m/s).

# Arguments
- `diff`          : 2D scalar field to display
- `show_colorbar` : Whether to add a horizontal colour bar (default: true)
(remaining arguments same as plot_colormesh_triple)
"""
function plot_colormesh(xi, yi, diff, mask, title_name, outfile_name; show_colorbar=true)
    fig = figure(figsize=(12, 12))
    ax  = subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")

    pc = ax.pcolormesh(xi, yi, diff, shading="auto",
                        cmap=ColorMap("RdBu_r"), vmin=-0.2, vmax=0.2,
                        transform=ccrs.PlateCarree(), zorder=1)

    if show_colorbar
        cb = colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label("m/s")
    end

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                transform=ccrs.PlateCarree(), zorder=1)
    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                transform=ccrs.PlateCarree(), zorder=2)
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false,
                 linewidth=0.5, alpha=0.5, linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_extent([1.2, 4.26, 40.5, 43.06])

    savefig(outfile_name)
    close()
end


"""
    plot_colormesh_vel_rel(xi, yi, diff, mask, title_name, outfile_name;
                           show_colorbar=true)

Creates a single-panel colour-mesh map of a relative velocity difference field
using a diverging RdBu_r colour map (range ±1, dimensionless).

Identical to `plot_colormesh` except the colour range is ±1 (suitable for
relative/normalised differences).
"""
function plot_colormesh_vel_rel(xi, yi, diff, mask, title_name, outfile_name; show_colorbar=true)
    fig = figure(figsize=(12, 12))
    ax  = subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")

    pc = ax.pcolormesh(xi, yi, diff, shading="auto",
                        cmap=ColorMap("RdBu_r"), vmin=-1., vmax=1.,
                        transform=ccrs.PlateCarree(), zorder=1)

    if show_colorbar
        cb = colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    end

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                transform=ccrs.PlateCarree(), zorder=1)
    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                transform=ccrs.PlateCarree(), zorder=2)
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false,
                 linewidth=0.5, alpha=0.5, linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_extent([1.2, 4.26, 40.5, 43.06])

    savefig(outfile_name)
    close()
end


"""
    plot_colormesh_std(xi, yi, diff, mask, title_name, outfile_name;
                       show_colorbar=true)

Creates a single-panel colour-mesh map of a standard deviation field
using a sequential viridis colour map (range 0–0.05 m/s).
"""
function plot_colormesh_std(xi, yi, diff, mask, title_name, outfile_name; show_colorbar=true)
    fig = figure(figsize=(12, 12))
    ax  = subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")

    pc = ax.pcolormesh(xi, yi, diff, shading="auto",
                        cmap=ColorMap("viridis"), vmin=0., vmax=0.05,
                        transform=ccrs.PlateCarree(), zorder=1)

    if show_colorbar
        cb = colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label("m/s")
    end

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper", alpha=1.0,
                transform=ccrs.PlateCarree(), zorder=1)
    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], cmap="Greys",  alpha=0.1,
                transform=ccrs.PlateCarree(), zorder=2)
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false,
                 linewidth=0.5, alpha=0.5, linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_extent([1.2, 4.26, 40.5, 43.06])

    savefig(outfile_name)
    close()
end

# ------------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------------

# --- ICATMAR bathymetry grid ---
file_icatmar = "../../data/bathy.nc"
mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar)

# ------------------------------------------------------------------------------------

# --- Least Squares (LS) output grid ---
grid_file = "../../data/hfradar_totals_grid_icatmar.nc"

lon_LS, lat_LS, lon_grid, lat_grid, mask_antena = processor.read_grid_from_netcdf(grid_file)

# Build the full 2D LS coordinate grids
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# ------------------------------------------------------------------------------------

# --- Copernicus MEDSEA model (ground truth) ---
file_cop = "../../data/january_2026/data_medsea/all_data_january_2026.nc"
# file_cop = "../../data/february_2025/data_medsea/all_data_february_2025.nc"

lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")

# Build meshgrid and convert from radians to degrees
lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:, :], lat_cop[:, :])
lon_mesh_model_cop = lon_mesh_model_cop .* (180 / pi)
lat_mesh_model_cop = lat_mesh_model_cop .* (180 / pi)

# Replace fill values (1e20) with NaN
u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)

println("COPERNICUS read")

# ------------------------------------------------------------------------------------

# --- DIVAnd outputs (radial and total reconstructions) ---

# Interpolate the land/sea mask to the LS grid for later filtering
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

divand_file = "../../data/january_2026/divand/divand_field.nc"
# divand_file = "../../data/february_2025/data_divand_10_days/divand_field.nc"

# Read all DIVAnd velocity fields from the NetCDF file in one block
data = NCDataset(divand_file, "r") do ds
    (
        lon      = ds["longitude"][:, :],
        lat      = ds["latitude"][:, :],
        time     = ds["time"][:],
        u_radial = ds["u_radial_divand"][:, :, :],
        v_radial = ds["v_radial_divand"][:, :, :],
        u_total  = ds["u_total_divand"][:, :, :],
        v_total  = ds["v_total_divand"][:, :, :]
    )
end

uri = data.u_radial   # DIVAnd east velocity from radials
vri = data.v_radial   # DIVAnd north velocity from radials
uti = data.u_total    # DIVAnd east velocity from totals
vti = data.v_total    # DIVAnd north velocity from totals

# ------------------------------------------------------------------------------------
# MAIN LOOP: COMPUTE DIFFERENCES AND STATISTICS FOR EACH SNAPSHOT
# ------------------------------------------------------------------------------------

n_times   = 264
grid_size = size(uri[1, :, :])   # Spatial size of the LS grid (n_lon × n_lat)

# Pre-allocate 3D arrays (time × lon × lat) for velocity differences
diff_rad_u_all   = Array{Float32}(undef, n_times, grid_size...)
diff_rad_v_all   = Array{Float32}(undef, n_times, grid_size...)
diff_rad_mag_all = Array{Float32}(undef, n_times, grid_size...)

diff_tot_u_all   = Array{Float32}(undef, n_times, grid_size...)
diff_tot_v_all   = Array{Float32}(undef, n_times, grid_size...)
diff_tot_mag_all = Array{Float32}(undef, n_times, grid_size...)

diff_LS_u_all    = Array{Float32}(undef, n_times, grid_size...)
diff_LS_v_all    = Array{Float32}(undef, n_times, grid_size...)
diff_LS_mag_all  = Array{Float32}(undef, n_times, grid_size...)

# Pre-allocate arrays for relative velocity magnitude differences
vel_rel_rad_mag_all = Array{Float32}(undef, n_times, grid_size...)
vel_rel_tot_mag_all = Array{Float32}(undef, n_times, grid_size...)
vel_rel_LS_mag_all  = Array{Float32}(undef, n_times, grid_size...)

# Arrays for time-series summary statistics
diff_n_points_valid_rad = zeros(n_times, 1)
diff_n_points_valid_tot = zeros(n_times, 1)

vel_time_series = zeros(n_times, 5)   # [i, avg_cop, avg_rad, avg_tot, avg_LS]
std_time_series = zeros(n_times, 5)   # [i, n_valid, std_rad, std_tot, std_LS]

for i = 1:n_times

    println("ITERATION ... ", i)

    # ── Copernicus snapshot at time step i ───────────────────────────
    u_cop_timestep = u_model_cop[i, :, :]
    v_cop_timestep = v_model_cop[i, :, :]

    # ── DIVAnd snapshots at time step i ─────────────────────────────
    uri_timestep = uri[i, :, :]   # Radial reconstruction – east component
    vri_timestep = vri[i, :, :]   # Radial reconstruction – north component
    uti_timestep = uti[i, :, :]   # Total reconstruction – east component
    vti_timestep = vti[i, :, :]   # Total reconstruction – north component

    # ── Least Squares total velocities for snapshot i ────────────────
    file_tuv = ("../../data/january_2026/totals_10_days/medsea_totals_"
                * lpad(string(i - 1), 3, '0') * "_all_grid.txt")
    # file_tuv = ("../../data/february_2025/totals_10_days/medsea_totals_"
    #             * lpad(string(i - 1), 3, '0') * "_all_grid.txt")

    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array     = df.longitude
    lat_array     = df.latitude
    u_array       = df.u_total
    v_array       = df.v_total
    mod_vel_array = df.modulo
    angle_array   = df.angulo
    gdop_array    = df.gdop

    # Filter out physically unrealistic or geometrically poorly constrained points
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)

    lon_array     = lon_array[valid_index]
    lat_array     = lat_array[valid_index]
    u_array       = u_array[valid_index]
    v_array       = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array   = angle_array[valid_index]

    println("TOTALS COP read")

    # Map the LS scattered points onto the 2D LS grid
    u_LS_2d, v_LS_2d = ma.putting_points_to_LS_grid(
        lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array
    )

    # ── Interpolate Copernicus ground truth to the LS grid ───────────
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(
        lon_mesh_model_cop, lat_mesh_model_cop,
        u_cop_timestep, v_cop_timestep,
        lon_grid_LS, lat_grid_LS,
        mask_coarse_grid=[], points_mode=false
    )

    # Build a mask of grid points that are valid in ALL methods simultaneously
    valid_mask = .!(
        isnan.(u_interp_cop) .| isnan.(v_interp_cop) .|
        isnan.(uri_timestep) .| isnan.(vri_timestep) .|
        isnan.(uti_timestep) .| isnan.(vti_timestep) .|
        isnan.(u_LS_2d)      .| isnan.(v_LS_2d)
    )

    # Extract only the commonly valid grid points
    u_interp_cop_filtered = u_interp_cop[valid_mask]
    v_interp_cop_filtered = v_interp_cop[valid_mask]
    uri_filtered          = uri_timestep[valid_mask]
    vri_filtered          = vri_timestep[valid_mask]
    uti_filtered          = uti_timestep[valid_mask]
    vti_filtered          = vti_timestep[valid_mask]
    u_array_filtered      = u_LS_2d[valid_mask]
    v_array_filtered      = v_LS_2d[valid_mask]

    # ── Velocity magnitudes ──────────────────────────────────────────
    vel_mag_cop = sqrt.(u_interp_cop_filtered .^ 2 .+ v_interp_cop_filtered .^ 2)
    vel_mag_rad = sqrt.(uri_filtered          .^ 2 .+ vri_filtered          .^ 2)
    vel_mag_tot = sqrt.(uti_filtered          .^ 2 .+ vti_filtered          .^ 2)
    vel_mag_LS  = sqrt.(u_array_filtered      .^ 2 .+ v_array_filtered      .^ 2)

    # ── Domain-averaged magnitudes ───────────────────────────────────
    avg_vel_cop = mean(vel_mag_cop)
    avg_vel_rad = mean(vel_mag_rad)
    avg_vel_tot = mean(vel_mag_tot)
    avg_vel_LS  = mean(vel_mag_LS)

    println("AVERAGES:")
    println("COP = ", avg_vel_cop)
    println("RAD = ", avg_vel_rad)
    println("TOT = ", avg_vel_tot)
    println("LS  = ", avg_vel_LS)

    vel_time_series[i, :] = [i, avg_vel_cop, avg_vel_rad, avg_vel_tot, avg_vel_LS]

    # ── Initialise difference arrays with NaN (spatial grid) ─────────
    diff_rad_u   = fill(NaN32, grid_size...)
    diff_rad_v   = fill(NaN32, grid_size...)
    diff_rad_mag = fill(NaN32, grid_size...)

    diff_tot_u   = fill(NaN32, grid_size...)
    diff_tot_v   = fill(NaN32, grid_size...)
    diff_tot_mag = fill(NaN32, grid_size...)

    diff_LS_u    = fill(NaN32, grid_size...)
    diff_LS_v    = fill(NaN32, grid_size...)
    diff_LS_mag  = fill(NaN32, grid_size...)

    vel_relative_rad = fill(NaN32, grid_size...)
    vel_relative_tot = fill(NaN32, grid_size...)
    vel_relative_LS  = fill(NaN32, grid_size...)

    # ── Velocity component and magnitude differences (method – Copernicus) ──
    diff_rad_u[valid_mask]   = uri_filtered     .- u_interp_cop_filtered
    diff_rad_v[valid_mask]   = vri_filtered     .- v_interp_cop_filtered
    diff_rad_mag[valid_mask] = vel_mag_rad      .- vel_mag_cop

    diff_tot_u[valid_mask]   = uti_filtered     .- u_interp_cop_filtered
    diff_tot_v[valid_mask]   = vti_filtered     .- v_interp_cop_filtered
    diff_tot_mag[valid_mask] = vel_mag_tot      .- vel_mag_cop

    diff_LS_u[valid_mask]    = u_array_filtered .- u_interp_cop_filtered
    diff_LS_v[valid_mask]    = v_array_filtered .- v_interp_cop_filtered
    diff_LS_mag[valid_mask]  = vel_mag_LS       .- vel_mag_cop

    # ── Relative velocity magnitude differences: (method – COP) / COP ──
    vel_relative_rad[valid_mask] = (vel_mag_rad .- vel_mag_cop) ./ vel_mag_cop
    vel_relative_tot[valid_mask] = (vel_mag_tot .- vel_mag_cop) ./ vel_mag_cop
    vel_relative_LS[valid_mask]  = (vel_mag_LS  .- vel_mag_cop) ./ vel_mag_cop

    # ── Store spatial difference fields for this time step ───────────
    diff_rad_u_all[i, :, :]   = diff_rad_u
    diff_rad_v_all[i, :, :]   = diff_rad_v
    diff_rad_mag_all[i, :, :] = diff_rad_mag

    diff_tot_u_all[i, :, :]   = diff_tot_u
    diff_tot_v_all[i, :, :]   = diff_tot_v
    diff_tot_mag_all[i, :, :] = diff_tot_mag

    diff_LS_u_all[i, :, :]    = diff_LS_u
    diff_LS_v_all[i, :, :]    = diff_LS_v
    diff_LS_mag_all[i, :, :]  = diff_LS_mag

    vel_rel_rad_mag_all[i, :, :] = vel_relative_rad
    vel_rel_tot_mag_all[i, :, :] = vel_relative_tot
    vel_rel_LS_mag_all[i, :, :]  = vel_relative_LS

    # ── Snapshot-level standard deviation of the magnitude differences ──
    avg_std_rad = mean(snapshot_nanstd(diff_rad_mag_all[i, :, :], 1))
    avg_std_tot = mean(snapshot_nanstd(diff_tot_mag_all[i, :, :], 1))
    avg_std_LS  = mean(snapshot_nanstd(diff_LS_mag_all[i, :, :],  1))

    println("STANDARD DEVIATION AVERAGES: ...")
    println("RAD = ... ", avg_std_rad)
    println("TOT = ... ", avg_std_tot)
    println("LS  = ... ", avg_std_LS)

    n_valid_points = sum(valid_mask)
    println("Valid points in this iteration: ", n_valid_points)

    std_time_series[i, :] = [i, n_valid_points, avg_std_rad, avg_std_tot, avg_std_LS]

end

# ------------------------------------------------------------------------------------
# TEMPORAL AVERAGING
# ------------------------------------------------------------------------------------

"""
    avg_temp(diff_vel) -> Array (2D)

Computes the temporal (time-dimension) mean of a 3D difference array,
ignoring NaN values. Returns a 2D spatial field.
"""
function avg_temp(diff_vel)
    return dropdims(
        mean(x -> isnan(x) ? zero(x) : x, diff_vel, dims=1), dims=1
    ) ./
    dropdims(
        mean(x -> isnan(x) ? 0 : 1,       diff_vel, dims=1), dims=1
    )
end

# Temporal mean of velocity component differences
diff_rad_u_mean   = avg_temp(diff_rad_u_all)
diff_rad_v_mean   = avg_temp(diff_rad_v_all)
diff_rad_mag_mean = avg_temp(diff_rad_mag_all)

diff_tot_u_mean   = avg_temp(diff_tot_u_all)
diff_tot_v_mean   = avg_temp(diff_tot_v_all)
diff_tot_mag_mean = avg_temp(diff_tot_mag_all)

diff_LS_u_mean    = avg_temp(diff_LS_u_all)
diff_LS_v_mean    = avg_temp(diff_LS_v_all)
diff_LS_mag_mean  = avg_temp(diff_LS_mag_all)

# Temporal mean of relative velocity magnitude differences
vel_rel_rad_mag_mean = avg_temp(vel_rel_rad_mag_all)
vel_rel_tot_mag_mean = avg_temp(vel_rel_tot_mag_all)
vel_rel_LS_mag_mean  = avg_temp(vel_rel_LS_mag_all)

# ------------------------------------------------------------------------------------
# SAVE TIME-SERIES STATISTICS TO TEXT FILES
# ------------------------------------------------------------------------------------

# output_file = "../../data/january_2026/stats_data/velocity_magnitude_times_series.txt"
output_file = "../../data/february_2025/stats_data/velocity_magnitude_times_series.txt"
writedlm(output_file, vel_time_series, ' ')
println("VEL file generated: ", output_file)

# output_file = "../../data/january_2026/stats_data/standard_deviation_diff_times_series.txt"
output_file = "../../data/february_2025/stats_data/standard_deviation_diff_times_series.txt"
writedlm(output_file, std_time_series, ' ')
println("STD file generated: ", output_file)

# ------------------------------------------------------------------------------------
# COMPUTE TEMPORAL STANDARD DEVIATIONS OF DIFFERENCES
# ------------------------------------------------------------------------------------

std_diff_rad_u   = nanstd(diff_rad_u_all,   1)
std_diff_rad_v   = nanstd(diff_rad_v_all,   1)
std_diff_rad_mag = nanstd(diff_rad_mag_all, 1)

std_diff_tot_u   = nanstd(diff_tot_u_all,   1)
std_diff_tot_v   = nanstd(diff_tot_v_all,   1)
std_diff_tot_mag = nanstd(diff_tot_mag_all, 1)

std_diff_LS_u    = nanstd(diff_LS_u_all,    1)
std_diff_LS_v    = nanstd(diff_LS_v_all,    1)
std_diff_LS_mag  = nanstd(diff_LS_mag_all,  1)

# Replace zeros (produced by all-NaN slices) with NaN so they do not distort plots
std_diff_rad_u[std_diff_rad_u .== 0]     .= NaN
std_diff_rad_v[std_diff_rad_v .== 0]     .= NaN
std_diff_rad_mag[std_diff_rad_mag .== 0] .= NaN

std_diff_tot_u[std_diff_tot_u .== 0]     .= NaN
std_diff_tot_v[std_diff_tot_v .== 0]     .= NaN
std_diff_tot_mag[std_diff_tot_mag .== 0] .= NaN

std_diff_LS_u[std_diff_LS_u .== 0]       .= NaN
std_diff_LS_v[std_diff_LS_v .== 0]       .= NaN
std_diff_LS_mag[std_diff_LS_mag .== 0]   .= NaN

# ------------------------------------------------------------------------------------
# SAVE COMPARISON STATISTICS TO NETCDF FILES
# ------------------------------------------------------------------------------------

"""
    create_netcdf_comparison(output_file, lon_grid, lat_grid,
                             data_rad, data_tot, data_LS; metric="diff")

Writes a comparison NetCDF file containing one variable per reconstruction
method (DIVAnd-radials, DIVAnd-totals, Least Squares) for a given metric.

# Arguments
- `output_file`        : Output NetCDF filepath
- `lon_grid, lat_grid` : 2D coordinate grids (LS grid)
- `data_rad`           : Spatial field for the DIVAnd-radials method
- `data_tot`           : Spatial field for the DIVAnd-totals method
- `data_LS`            : Spatial field for the Least Squares method
- `metric`             : Either "diff" (mean difference) or "std" (standard deviation)
"""
function create_netcdf_comparison(output_file, lon_grid, lat_grid,
                                   data_rad, data_tot, data_LS; metric="diff")

    # Set variable naming and metadata based on the requested metric
    if metric == "diff"
        var_prefix            = "diff"
        var_suffix            = "_mag_mean"
        long_name_prefix      = "Mean speed anomaly"
        description_prefix    = "Temporal mean of speed magnitude difference"
        title                 = "Mean velocity magnitude differences - Comparison of reconstruction methods"
    elseif metric == "std"
        var_prefix            = "std_diff"
        var_suffix            = "_mag"
        long_name_prefix      = "Standard deviation of mean anomaly"
        description_prefix    = "Standard deviation of speed magnitude difference"
        title                 = "Standard deviation of velocity magnitude differences - Comparison of reconstruction methods"
    else
        error("metric must be 'diff' or 'std'")
    end

    NCDataset(output_file, "c") do ds

        # Define spatial dimensions
        defDim(ds, "lon", size(lon_grid, 1))
        defDim(ds, "lat", size(lon_grid, 2))

        # Coordinate variables
        defVar(ds, "longitude", lon_grid, ("lon", "lat"), attrib=Dict(
            "long_name"     => "Longitude",
            "units"         => "degrees_east",
            "standard_name" => "longitude"
        ))
        defVar(ds, "latitude",  lat_grid, ("lon", "lat"), attrib=Dict(
            "long_name"     => "Latitude",
            "units"         => "degrees_north",
            "standard_name" => "latitude"
        ))

        # DIVAnd-radials variable
        defVar(ds, var_prefix * "_rad" * var_suffix, data_rad, ("lon", "lat"), attrib=Dict(
            "long_name"   => "$long_name_prefix (DIVAnd-radials vs MEDSEA)",
            "units"       => "m/s",
            "description" => "$description_prefix between DIVAnd reconstruction from radials and MEDSEA model"
        ))

        # DIVAnd-totals variable
        defVar(ds, var_prefix * "_tot" * var_suffix, data_tot, ("lon", "lat"), attrib=Dict(
            "long_name"   => "$long_name_prefix (DIVAnd-totals vs MEDSEA)",
            "units"       => "m/s",
            "description" => "$description_prefix between DIVAnd reconstruction from totals and MEDSEA model"
        ))

        # Least Squares variable
        defVar(ds, var_prefix * "_LS" * var_suffix,  data_LS,  ("lon", "lat"), attrib=Dict(
            "long_name"   => "$long_name_prefix (LS vs MEDSEA)",
            "units"       => "m/s",
            "description" => "$description_prefix between Least Squares reconstruction and MEDSEA model"
        ))

        # Global attributes (CF-1.6 compliant)
        ds.attrib["title"]       = title
        ds.attrib["institution"] = "ICM-CSIC"
        ds.attrib["source"]      = "HF Radar data comparison with MEDSEA model"
        ds.attrib["history"]     = "Created on $(now())"
        ds.attrib["references"]  = "Catalan Sea HF Radar network"
        ds.attrib["Conventions"] = "CF-1.6"
    end

    println("NetCDF file created: $output_file")
end


# --- Save temporal mean differences ---
create_netcdf_comparison(
    "../../data/january_2026/stats_data/matrices_diff_mean.nc",
    lon_grid_LS, lat_grid_LS,
    diff_rad_mag_mean, diff_tot_mag_mean, diff_LS_mag_mean,
    metric="diff"
)

# --- Save temporal standard deviations ---
create_netcdf_comparison(
    "../../data/january_2026/stats_data/matrices_std_mean.nc",
    lon_grid_LS, lat_grid_LS,
    std_diff_rad_mag, std_diff_tot_mag, std_diff_LS_mag,
    metric="std"
)
