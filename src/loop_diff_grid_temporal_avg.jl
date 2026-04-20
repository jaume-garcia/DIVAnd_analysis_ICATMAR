"""
Grid-based temporal average and standard deviation analysis.
Compares DIVAnd (radial/total) and LS totals against Copernicus ground truth
over the full domain, computing mean differences and temporal variability.
"""

using PyCall
using PyPlot
using CSV, DataFrames
using DelimitedFiles
using NCDatasets

# Include Julia modules
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")

# Add Python script directory
push!(PyVector(pyimport("sys")."path"), "../utils")

# Import Python modules
hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")
ve = pyimport("vel_eta")

# Initialize HF Radar processor
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------
bathy_file = "../data/bathy.nc"
mask, h, pm, pn, xi, yi = grid_icatmar(bathy_file)

# ------------------------------------------------------------------------------------
# LS GRID
# ------------------------------------------------------------------------------------

grid_file = "../data/hfradar_totals_grid_icatmar.nc"
lon_LS, lat_LS, lon_grid, lat_grid, antenna_mask = processor.read_grid_from_netcdf(grid_file)
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# Group 1: BEGU, CREU, TOSS coverage
mask_group1 = dropdims(sum(antenna_mask[2:5, :, :], dims=1), dims=1)

# Group 2: AREN, GNST, PBCN coverage
mask_group2 = dropdims(sum(antenna_mask[6:8, :, :], dims=1), dims=1)

# Total coverage mask (union of all radars)
total_mask = mask_group1 + mask_group2
total_mask[total_mask .> 1] .= 1

total_mask_t = total_mask'
total_mask_t_float = Float64.(total_mask_t)
replace!(total_mask_t_float, 0.0 => NaN)
total_mask_t = total_mask_t_float

println("SIZE TOTAL MASK = ", size(total_mask_t))

# ------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ------------------------------------------------------------------------------------

function plot_colormesh(xi, yi, diff, mask, title_name, output_file; show_colorbar=true)
    """
    Plot 2D colormesh of difference fields.
    """
    fig = figure(figsize=(9, 9))
    ax = subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    
    pc = ax.pcolormesh(xi, yi, diff, shading="auto", cmap=ColorMap("RdYlBu"),
                       vmin=-0.2, vmax=0.2, transform=ccrs.PlateCarree(), zorder=0)
    
    if show_colorbar
        cb = colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label("m/s")
    end
    
    land_mask = mask .<= 0.5
    ax.contourf(xi, yi, land_mask, levels=[0.5, 1.0], cmap="copper", alpha=1.0, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false, linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_extent([1.2, 4.26, 40.5, 43.06])
    savefig(output_file)
    close()
end

function plot_colormesh_std(xi, yi, diff, mask, title_name, output_file; show_colorbar=true)
    """
    Plot 2D colormesh of standard deviation fields.
    """
    fig = figure(figsize=(9, 9))
    ax = subplot(111, projection=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    
    pc = ax.pcolormesh(xi, yi, diff, shading="auto", cmap=ColorMap("viridis"),
                       vmin=0., vmax=0.05, transform=ccrs.PlateCarree())
    
    if show_colorbar
        cb = colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label("m/s")
    end
    
    land_mask = mask .<= 0.5
    ax.contourf(xi, yi, land_mask, levels=[0.5, 1.0], cmap="copper", alpha=1.0, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false, linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_extent([1.2, 4.26, 40.5, 43.06])
    savefig(output_file)
    close()
end

# ------------------------------------------------------------------------------------
# COPERNICUS GROUND TRUTH
# ------------------------------------------------------------------------------------

copernicus_file = "../data/january_2026/data_medsea/all_data_january_2026.nc"

@time lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(copernicus_file, "lon", "lat", "u_data", "v_data", "0:")

lon_mesh_cop, lat_mesh_cop = meshgrid(lon_cop[:, :], lat_cop[:, :])
lon_mesh_cop = lon_mesh_cop .* (180/π)
lat_mesh_cop = lat_mesh_cop .* (180/π)

u_cop_model = replace(u_cop, 1.0f20 => NaN32)
v_cop_model = replace(v_cop, 1.0f20 => NaN32)

println("COPERNICUS data loaded")

# ------------------------------------------------------------------------------------
# INTERPOLATE BATHYMETRY MASK TO LS GRID
# ------------------------------------------------------------------------------------

mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

# ------------------------------------------------------------------------------------
# DIVAND RECONSTRUCTION DATA
# ------------------------------------------------------------------------------------

divand_file = "../data/january_2026/divand/divand_field.nc"

data = NCDataset(divand_file, "r") do ds
    (
        lon = ds["longitude"][:, :],
        lat = ds["latitude"][:, :],
        time = ds["time"][:],
        u_radial = ds["u_radial_divand"][:, :, :],
        v_radial = ds["v_radial_divand"][:, :, :],
        u_total = ds["u_total_divand"][:, :, :],
        v_total = ds["v_total_divand"][:, :, :]
    )
end

u_radial = data.u_radial
v_radial = data.v_radial
u_total = data.u_total
v_total = data.v_total

# ------------------------------------------------------------------------------------
# INITIALIZE 3D ARRAYS FOR DIFFERENCES
# ------------------------------------------------------------------------------------

n_times = 264
grid_size = size(u_radial[1, :, :])

# Radial method differences
diff_rad_u_all = Array{Float32}(undef, n_times, grid_size...)
diff_rad_v_all = Array{Float32}(undef, n_times, grid_size...)
diff_rad_speed_all = Array{Float32}(undef, n_times, grid_size...)

# Total method differences
diff_total_u_all = Array{Float32}(undef, n_times, grid_size...)
diff_total_v_all = Array{Float32}(undef, n_times, grid_size...)
diff_total_speed_all = Array{Float32}(undef, n_times, grid_size...)

# Time series of domain-averaged speeds
speed_time_series = zeros(n_times, 6)

# ------------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------------

for i = 1:n_times

    println("ITERATION ... ", i)

    # Copernicus snapshot
    u_cop_ts = u_cop_model[i, :, :]
    v_cop_ts = v_cop_model[i, :, :]
    
    # DIVAnd velocities
    u_radial_ts = u_radial[i, :, :]
    v_radial_ts = v_radial[i, :, :]
    u_total_ts = u_total[i, :, :]
    v_total_ts = v_total[i, :, :]
    
    # LS total velocities (Level 3)
    totals_file = "../data/january_2026/totals_10_days/medsea_totals_" * lpad(string(i-1), 3, '0') * "_all_grid.txt"

    df = CSV.read(totals_file, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = df.u_total
    v_array = df.v_total
    speed_array = df.speed
    angle_array = df.direction
    gdop_array = df.gdop
    
    # Filter by GDOP and maximum speed
    valid_idx = (gdop_array .<= 2.0) .& (speed_array .<= 1.2)

    lon_array = lon_array[valid_idx]
    lat_array = lat_array[valid_idx]
    u_array = u_array[valid_idx]
    v_array = v_array[valid_idx]
    speed_array = speed_array[valid_idx]
    angle_array = angle_array[valid_idx]
    
    # Grid LS velocities
    u_LS_2d, v_LS_2d = ve.putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array)
    
    println("LS totals processed")
    
    # Interpolate Copernicus to LS grid
    u_cop_interp, v_cop_interp = ma.interp_grid_vel(
        lon_mesh_cop, lat_mesh_cop, u_cop_ts, v_cop_ts,
        lon_grid_LS, lat_grid_LS, mask_coarse_grid=[], points_mode=false
    )
    
    # Speed magnitudes
    speed_cop = sqrt.(u_cop_interp.^2 .+ v_cop_interp.^2)
    speed_rad = sqrt.(u_radial_ts.^2 .+ v_radial_ts.^2)
    speed_total = sqrt.(u_total_ts.^2 .+ v_total_ts.^2)
    speed_LS = sqrt.(u_LS_2d.^2 .+ v_LS_2d.^2)
    
    # Copernicus speed masked to theoretical radar coverage
    speed_cop_masked = speed_cop .* total_mask_t
    
    # Valid mask (no NaNs for all fields)
    valid_mask = .!isnan.(speed_cop) .& .!isnan.(speed_rad) .& 
                 .!isnan.(speed_total) .& .!isnan.(speed_cop_masked) .& .!isnan.(speed_LS)
    
    # Clean arrays (only valid points)
    speed_cop_clean = speed_cop[valid_mask]
    speed_rad_clean = speed_rad[valid_mask]
    speed_total_clean = speed_total[valid_mask]
    speed_LS_clean = speed_LS[valid_mask]
    speed_cop_masked_clean = speed_cop_masked[.!isnan.(speed_cop_masked)]
    
    # Domain-averaged speeds
    avg_speed_cop = mean(speed_cop_clean)
    avg_speed_rad = mean(speed_rad_clean)
    avg_speed_total = mean(speed_total_clean)
    avg_speed_LS = mean(speed_LS_clean)
    avg_speed_cop_masked = mean(speed_cop_masked_clean)
    
    speed_time_series[i, :] = [i, avg_speed_cop, avg_speed_cop_masked, avg_speed_rad, avg_speed_total, avg_speed_LS]
    
    # Store differences in 3D arrays
    diff_rad_u_all[i, :, :] = u_radial_ts .- u_cop_interp
    diff_rad_v_all[i, :, :] = v_radial_ts .- v_cop_interp
    diff_rad_speed_all[i, :, :] = speed_rad .- speed_cop
    
    diff_total_u_all[i, :, :] = u_total_ts .- u_cop_interp
    diff_total_v_all[i, :, :] = v_total_ts .- v_cop_interp
    diff_total_speed_all[i, :, :] = speed_total .- speed_cop
end

# ------------------------------------------------------------------------------------
# COMPUTE TEMPORAL MEANS (ignoring NaNs)
# ------------------------------------------------------------------------------------

# Helper function to compute mean ignoring NaNs
function temporal_mean(arr, dim)
    sum_val = dropdims(sum(x -> isnan(x) ? 0.0 : x, arr, dims=dim), dims=dim)
    count_val = dropdims(sum(x -> isnan(x) ? 0 : 1, arr, dims=dim), dims=dim)
    return sum_val ./ count_val
end

# Radial method mean differences
diff_rad_u_mean = temporal_mean(diff_rad_u_all, 1) .* total_mask_t
diff_rad_v_mean = temporal_mean(diff_rad_v_all, 1) .* total_mask_t
diff_rad_speed_mean = temporal_mean(diff_rad_speed_all, 1) .* total_mask_t

# Total method mean differences
diff_total_u_mean = temporal_mean(diff_total_u_all, 1) .* total_mask_t
diff_total_v_mean = temporal_mean(diff_total_v_all, 1) .* total_mask_t
diff_total_speed_mean = temporal_mean(diff_total_speed_all, 1) .* total_mask_t

println("\nTemporal means calculated (ignoring NaNs)")
println("Mean difference matrix size: ", size(diff_rad_u_mean))
println("Speed time series matrix size: ", size(speed_time_series))

# Save speed time series
output_file = "../data/january_2026/stats_data/speed_magnitude_time_series.txt"
writedlm(output_file, speed_time_series, ' ')
println("ASCII file generated: ", output_file)

# ------------------------------------------------------------------------------------
# COMPUTE TEMPORAL STANDARD DEVIATIONS (ignoring NaNs)
# ------------------------------------------------------------------------------------

function temporal_std(arr, dim)
    """
    Calculate temporal standard deviation ignoring NaNs.
    """
    # Mean ignoring NaNs
    μ = temporal_mean(arr, dim)
    
    # Expand dimensions for broadcasting
    μ_expanded = reshape(μ, (1, size(μ)...))
    
    # Squared differences
    squared_diff = (arr .- μ_expanded).^2
    
    # Variance (mean of squared differences ignoring NaNs)
    variance = temporal_mean(squared_diff, dim)
    
    return sqrt.(variance)
end

# Radial method standard deviations
std_diff_rad_u = temporal_std(diff_rad_u_all, 1) .* total_mask_t
std_diff_rad_v = temporal_std(diff_rad_v_all, 1) .* total_mask_t
std_diff_rad_speed = temporal_std(diff_rad_speed_all, 1) .* total_mask_t

# Total method standard deviations
std_diff_total_u = temporal_std(diff_total_u_all, 1) .* total_mask_t
std_diff_total_v = temporal_std(diff_total_v_all, 1) .* total_mask_t
std_diff_total_speed = temporal_std(diff_total_speed_all, 1) .* total_mask_t

println("\nTemporal standard deviations calculated (ignoring NaNs)")
println("Standard deviation matrix size: ", size(std_diff_rad_u))
