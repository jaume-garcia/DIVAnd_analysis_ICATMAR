using PyCall
using CSV, DataFrames
using DelimitedFiles
using NCDatasets
using Random
using Statistics
using Printf

# Julia scripts
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")

# Add Python script directory to Python path
push!(PyVector(pyimport("sys")."path"), "../utils")

# Import Python scripts
hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# HFRadarProcessor instance
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# LS grid
grid_file = "../data/hfradar_totals_grid_icatmar.nc"
lon_LS, lat_LS, lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# ------------------------------------------------------------------------------------
# COPERNICUS GROUND TRUTH
file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"
@time lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")

lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:,:], lat_cop[:,:])
lon_mesh_model_cop = lon_mesh_model_cop .*(180/pi)
lat_mesh_model_cop = lat_mesh_model_cop .*(180/pi)

u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)
println("COPERNICUS read")

# ------------------------------------------------------------------------------------
# DIVAND DATA
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

uri = data.u_radial   # u radial from DIVAnd
vri = data.v_radial   # v radial from DIVAnd
uti = data.u_total    # u total from DIVAnd
vti = data.v_total    # v total from DIVAnd

# Initialize results matrix: columns = [timestep, RMSE_LS, RMSE_radial, RMSE_total]
results_matrix = zeros(264, 4)

# ------------------------------------------------------------------------------------
# MAIN LOOP OVER TIME STEPS
# ------------------------------------------------------------------------------------

for i = 1:264

    println("ITERATION = ... ", i)

    # COPERNICUS SNAPSHOT VELOCITY
    u_cop_timestep = u_model_cop[i, :, :]
    v_cop_timestep = v_model_cop[i, :, :]
    
    # DIVAND VELOCITIES (RADIAL AND TOTAL)
    uri_timestep = uri[i, :, :]
    vri_timestep = vri[i, :, :]
    uti_timestep = uti[i, :, :]
    vti_timestep = vti[i, :, :]
    
    # TOTAL VELOCITIES (L3) FROM LS
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_"*lpad(string(i-1), 3, '0')*"_all_grid.txt"
    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = df.u_total 
    v_array = df.v_total 
    mod_vel_array = df.speed 
    angle_array = df.drection
    gdop_array = df.gdop
    
    # Filter outliers and non-physical velocities
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)
    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array = angle_array[valid_index]

    println("LENGTH TOTAL VELOCITIES L3 WITH JAUME LS (NO RESTRICTIONS) = ... ", length(u_array))
    println("TOTALS COP read")

    # INTERPOLATION (Copernicus ground truth to LS grid)
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop,
                                                     u_cop_timestep, v_cop_timestep,
                                                     lon_grid_LS, lat_grid_LS,
                                                     mask_coarse_grid=[], points_mode=false)
    
    println("SIZE COP = ", size(u_interp_cop))
    println("SIZE DV TOT = ", size(uri_timestep))
    println("SIZE DV RAD = ", size(uti_timestep))
    
    # Map LS points to LS grid
    u_LS_2d, v_LS_2d = ma.putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array)
    
    # Filter valid points (without NaNs)
    valid_mask = .!(isnan.(u_interp_cop) .| isnan.(v_interp_cop) .|
                isnan.(uri_timestep) .| isnan.(vri_timestep) .|
                isnan.(uti_timestep) .| isnan.(vti_timestep) .|
                isnan.(u_LS_2d) .| isnan.(v_LS_2d))
    
    # Filtered vectors for RMSE calculation
    u_interp_cop_filtered = u_interp_cop[valid_mask]
    v_interp_cop_filtered = v_interp_cop[valid_mask]
    uri_filtered = uri_timestep[valid_mask]
    vri_filtered = vri_timestep[valid_mask]
    uti_filtered = uti_timestep[valid_mask]
    vti_filtered = vti_timestep[valid_mask]
    u_array_filtered = u_LS_2d[valid_mask]
    v_array_filtered = v_LS_2d[valid_mask]
    
    N_obs = sum(valid_mask)
    println("Number of valid points for RMSE: $N_obs")

    # RMSE calculation for each method

    # 1. Least Squares (LS)
    total_rmse_LS = sqrt(sum(skipmissing((u_array_filtered .- u_interp_cop_filtered).^2 .+
                          (v_array_filtered .- v_interp_cop_filtered).^2)) / N_obs)

    # 2. DIVAnd Radials
    total_rmse_rad = sqrt(sum(skipmissing((uri_filtered .- u_interp_cop_filtered).^2 .+
                           (vri_filtered .- v_interp_cop_filtered).^2)) / N_obs)

    # 3. DIVAnd Totals
    total_rmse_tot = sqrt(sum(skipmissing((uti_filtered .- u_interp_cop_filtered).^2 .+
                           (vti_filtered .- v_interp_cop_filtered).^2)) / N_obs)

    println("\n=== RMSE RESULTS ===")
    println("Calculating RMSE = ... LS: $total_rmse_LS | RAD: $total_rmse_rad | TOT: $total_rmse_tot")

    # Store results
    results_matrix[i, :] = [i-1, total_rmse_LS, total_rmse_rad, total_rmse_tot]
end

# Save main results file
output_file = "../data/january_2026/stats_data/all_rms_grid_icatmar.txt"
writedlm(output_file, results_matrix, ' ')
println("ASCII file generated: ", output_file)

# ------------------------------------------------------------------------------------
# BOOTSTRAP FUNCTIONS
# ------------------------------------------------------------------------------------

"""
Block bootstrap for time series.
- rmse_t: vector of RMSE per snapshot
- B: number of bootstrap iterations
- block_size: temporal block size (not used in simple bootstrap)
"""
function bootstrap_rmse(rmse_t; B=5000)
    n = length(rmse_t)
    rmse_boot = zeros(B)

    for b in 1:B
        sample = rand(rmse_t, n)  # resampling with replacement
        rmse_boot[b] = sqrt(mean(sample.^2))
    end

    rmse_global = sqrt(mean(rmse_t.^2))
    se = std(rmse_boot)
    ci = quantile(rmse_boot, [0.025, 0.975])

    return rmse_global, se, ci
end

# ------------------------------------------------------------------------------------
# EXTRACT DATA
# ------------------------------------------------------------------------------------

rmse_LS  = results_matrix[:, 2]
rmse_rad = results_matrix[:, 3]
rmse_tot = results_matrix[:, 4]

# ------------------------------------------------------------------------------------
# BOOTSTRAP PARAMETERS
# ------------------------------------------------------------------------------------

B = 5000          # iterations
block_size = 10   # adjustable (see note below)

println("\nCalculating bootstrap...\n")

# ------------------------------------------------------------------------------------
# BOOTSTRAP CALCULATION
# ------------------------------------------------------------------------------------

rmseLS,  seLS,  ciLS  = bootstrap_rmse(rmse_LS;  B=B)
rmseRAD, seRAD, ciRAD = bootstrap_rmse(rmse_rad; B=B)
rmseTOT, seTOT, ciTOT = bootstrap_rmse(rmse_tot; B=B)

# ------------------------------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------------------------------

println("\n", "="^75)
println("BOOTSTRAP RMSE (with uncertainty)")
println("="^75)

println(@sprintf("%-20s  %-12s  %-12s  %-25s", "Method", "RMSE", "SE", "CI 95%"))
println("-"^75)

println(@sprintf("%-20s  %-12.6f  %-12.6f  [%.6f, %.6f]",
    "LS", rmseLS, seLS, ciLS[1], ciLS[2]))

println(@sprintf("%-20s  %-12.6f  %-12.6f  [%.6f, %.6f]",
    "DIVAnd Radials", rmseRAD, seRAD, ciRAD[1], ciRAD[2]))

println(@sprintf("%-20s  %-12.6f  %-12.6f  [%.6f, %.6f]",
    "DIVAnd Totals", rmseTOT, seTOT, ciTOT[1], ciTOT[2]))

println("="^75)

# ------------------------------------------------------------------------------------
# SAVE BOOTSTRAP RESULTS
# ------------------------------------------------------------------------------------

output_bootstrap = "../data/january_2026/stats_data/bootstrap_rmse_results.txt"

open(output_bootstrap, "w") do io
    println(io, "Method RMSE SE CI_low CI_high")
    println(io, "LS $rmseLS $seLS $(ciLS[1]) $(ciLS[2])")
    println(io, "Radials $rmseRAD $seRAD $(ciRAD[1]) $(ciRAD[2])")
    println(io, "Totals $rmseTOT $seTOT $(ciTOT[1]) $(ciTOT[2])")
end

println("\nBootstrap file saved at: ", output_bootstrap)

# ------------------------------------------------------------------------------------
# GLOBAL STATISTICS: MEAN AND STANDARD DEVIATION OF RMSE
# ------------------------------------------------------------------------------------

rmse_LS  = results_matrix[:, 2]
rmse_rad = results_matrix[:, 3]
rmse_tot = results_matrix[:, 4]

mean_rmse_LS  = mean(rmse_LS)
mean_rmse_rad = mean(rmse_rad)
mean_rmse_tot = mean(rmse_tot)

std_rmse_LS  = std(rmse_LS)
std_rmse_rad = std(rmse_rad)
std_rmse_tot = std(rmse_tot)

println("\n", "="^60)
println("GLOBAL RMSE STATISTICS")
println("="^60)
println(@sprintf("%-20s  %-18s  %-18s", "Method", "Mean RMSE (m/s)", "Std RMSE (m/s)"))
println("-"^60)
println(@sprintf("%-20s  %-18.6f  %-18.6f", "LS",             mean_rmse_LS,  std_rmse_LS))
println(@sprintf("%-20s  %-18.6f  %-18.6f", "DIVAnd Radials", mean_rmse_rad, std_rmse_rad))
println(@sprintf("%-20s  %-18.6f  %-18.6f", "DIVAnd Totals",  mean_rmse_tot, std_rmse_tot))
println("="^60)
