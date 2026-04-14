using PyCall
using CSV, DataFrames, NCDatasets, DelimitedFiles

# Include necessary Julia files
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

# CATALAN RADARS
radar_dict = hf.radars_cat()  

# ------------------------------------------------------------------------------------
# BATHYMETRY
file_icatmar = "../data/bathy.nc"
mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar)

# ------------------------------------------------------------------------------------
# LS grid
grid_file = "../data/hfradar_totals_grid_icatmar.nc"
lon_LS, lat_LS, lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# ------------------------------------------------------------------------------------
# COPERNICUS GROUND TRUTH
file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"
lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")
lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop, lat_cop)
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

# ------------------------------------------------------------------------------------
# TARGET COORDINATES (choose one)

# Small gap snapshot 0
#closest_lon = 3.491500
#closest_lat = 41.529099

# Large gap snapshot 0
closest_lon = 3.350140
closest_lat = 41.502102

# ------------------------------------------------------------------------------------
# NEIGHBORS: find the 4 closest points (up, down, left, right) on LS grid
# ------------------------------------------------------------------------------------

# Find index of central point in 1D lon/lat vectors
i_lon_center = argmin(abs.(lon_LS .- closest_lon))
i_lat_center = argmin(abs.(lat_LS .- closest_lat))

# Calculate grid spacing (assuming regular grid)
dlon = lon_LS[2] - lon_LS[1]
dlat = lat_LS[2] - lat_LS[1]

# Coordinates of the 4 neighbors
neighbor_labels = ["right", "left", "up", "down"]
neighbor_lons   = [lon_LS[i_lon_center] + dlon,   # right (+lon)
                   lon_LS[i_lon_center] - dlon,   # left (-lon)
                   lon_LS[i_lon_center],           # up (same lon)
                   lon_LS[i_lon_center]]           # down (same lon)
neighbor_lats   = [lat_LS[i_lat_center],           # right
                   lat_LS[i_lat_center],           # left
                   lat_LS[i_lat_center] + dlat,   # up (+lat)
                   lat_LS[i_lat_center] - dlat]   # down (-lat)

println("\nCentral point: lon=", closest_lon, "  lat=", closest_lat)
println("Central point indices on LS grid: i_lon=", i_lon_center, "  i_lat=", i_lat_center)
println("Grid spacing: dlon=", dlon, "  dlat=", dlat)
for k in 1:4
    println("  Neighbor ", neighbor_labels[k], ": lon=", neighbor_lons[k], "  lat=", neighbor_lats[k])
end

# ------------------------------------------------------------------------------------
# INITIALIZE ARRAYS FOR STORING RESULTS
# ------------------------------------------------------------------------------------

vel_mag_cop_array = zeros(264)  # Copernicus velocity magnitude
vel_mag_rad_array = zeros(264)  # DIVAnd radial velocity magnitude
vel_mag_tot_array = zeros(264)  # DIVAnd total velocity magnitude
vel_mag_LS_array  = zeros(264)  # Least Squares velocity magnitude

results_matrix = zeros(264, 13)  # Main results matrix

# Neighbors matrix: rows = timesteps, columns = 1 (timestep) + 4 neighbors * 12 = 49 columns
# For each neighbor: u_cop, v_cop, mod_cop, u_rad, v_rad, mod_rad, u_tot, v_tot, mod_tot, u_LS, v_LS, mod_LS
neighbors_matrix = fill(NaN, 264, 49)

# ------------------------------------------------------------------------------------
# MAIN LOOP OVER TIME STEPS
# ------------------------------------------------------------------------------------

for i = 1:264
    
    println("ITERATION NUMBER = ... ", i-1)
    u_cop_timestep = u_model_cop[i, :, :]
    v_cop_timestep = v_model_cop[i, :, :]
    
    # DIVAND VELOCITIES (RADIAL AND TOTAL)
    uri_timestep = uri[i, :, :]
    vri_timestep = vri[i, :, :]
    uti_timestep = uti[i, :, :]
    vti_timestep = vti[i, :, :]

    # ------------------------------------------------------------------------------------
    # TOTAL VELOCITIES (L3) FROM LS
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_"*lpad(string(i-1), 3, '0')*"_all_grid.txt"
    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array   = df.u_total
    v_array   = df.v_total 
    mod_vel_array = df.modulo 
    angle_array = df.angulo
    gdop_array  = df.gdop
   
    # Filter outliers and non-physical velocities
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)
    lon_array     = lon_array[valid_index]
    lat_array     = lat_array[valid_index]
    u_array       = u_array[valid_index]
    v_array       = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array   = angle_array[valid_index]

    # ------------------------------------------------------------------------------------
    # INTERPOLATE DIVAnd DATA TO OBSERVATION POINT
    
    # Radials
    interpolated_u_rad, interpolated_v_rad = ma.interp_grid_vel(lon_grid_LS, lat_grid_LS, uri_timestep, vri_timestep,
                                                                 [closest_lon], [closest_lat],
                                                                 mask_coarse_grid=[], points_mode=true)
    
    # Totals
    interpolated_u_tot, interpolated_v_tot = ma.interp_grid_vel(lon_grid_LS, lat_grid_LS, uti_timestep, vti_timestep,
                                                                 [closest_lon], [closest_lat],
                                                                 mask_coarse_grid=[], points_mode=true)
    
    # LS — exact match search in total velocities file
    tolerance = 1e-10
    exact_match = findall(
        (abs.(lon_array .- closest_lon) .< tolerance) .& 
        (abs.(lat_array .- closest_lat) .< tolerance)
    )
    
    if length(exact_match) > 0
        idx = exact_match[1]
        interpolated_u_LS = u_array[idx]
        interpolated_v_LS = v_array[idx]
    else
        interpolated_u_LS = NaN
        interpolated_v_LS = NaN
    end

    # Interpolate Copernicus data to observation point
    cop_interp_u, cop_interp_v = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop,
                                                     u_cop_timestep, v_cop_timestep,
                                                     [closest_lon], [closest_lat],
                                                     mask_coarse_grid=[], points_mode=true)
    
    # Calculate velocity magnitudes
    vel_mag_cop          = sqrt(cop_interp_u[1]^2 + cop_interp_v[1]^2)
    vel_mag_interp_rad   = sqrt(interpolated_u_rad[1]^2 + interpolated_v_rad[1]^2)
    vel_mag_interp_tot   = sqrt(interpolated_u_tot[1]^2 + interpolated_v_tot[1]^2)
    vel_mag_interp_LS    = sqrt(interpolated_u_LS[1]^2 + interpolated_v_LS[1]^2)
    
    # Store in arrays
    vel_mag_cop_array[i] = vel_mag_cop
    vel_mag_rad_array[i] = vel_mag_interp_rad
    vel_mag_tot_array[i] = vel_mag_interp_tot
    vel_mag_LS_array[i]  = vel_mag_interp_LS
    
    # Store in results matrix
    results_matrix[i, :] = [i-1, cop_interp_u[1], interpolated_u_rad[1], interpolated_u_tot[1], interpolated_u_LS[1],
                             cop_interp_v[1], interpolated_v_rad[1], interpolated_v_tot[1], interpolated_v_LS[1],
                             vel_mag_cop, vel_mag_interp_rad, vel_mag_interp_tot, vel_mag_interp_LS]

    # ------------------------------------------------------------------------------------
    # NEIGHBORS: interpolate Copernicus, DIVAnd radial/total, and exact match for LS
    # ------------------------------------------------------------------------------------

    neighbors_matrix[i, 1] = i - 1   # timestep (0-based)

    for k in 1:4
        n_lon = neighbor_lons[k]
        n_lat = neighbor_lats[k]

        # --- Copernicus: interpolation ---
        n_cop_u, n_cop_v = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop,
                                               u_cop_timestep, v_cop_timestep,
                                               [n_lon], [n_lat],
                                               mask_coarse_grid=[], points_mode=true)
        n_u_cop   = n_cop_u[1]
        n_v_cop   = n_cop_v[1]
        n_mod_cop = sqrt(n_u_cop^2 + n_v_cop^2)

        # --- DIVAnd Radial: interpolation ---
        n_rad_u, n_rad_v = ma.interp_grid_vel(lon_grid_LS, lat_grid_LS,
                                               uri_timestep, vri_timestep,
                                               [n_lon], [n_lat],
                                               mask_coarse_grid=[], points_mode=true)
        n_u_rad   = n_rad_u[1]
        n_v_rad   = n_rad_v[1]
        n_mod_rad = sqrt(n_u_rad^2 + n_v_rad^2)

        # --- DIVAnd Total: interpolation ---
        n_tot_u, n_tot_v = ma.interp_grid_vel(lon_grid_LS, lat_grid_LS,
                                               uti_timestep, vti_timestep,
                                               [n_lon], [n_lat],
                                               mask_coarse_grid=[], points_mode=true)
        n_u_tot   = n_tot_u[1]
        n_v_tot   = n_tot_v[1]
        n_mod_tot = sqrt(n_u_tot^2 + n_v_tot^2)

        # --- LS: exact match in total velocities file ---
        n_match = findall(
            (abs.(lon_array .- n_lon) .< tolerance) .& 
            (abs.(lat_array .- n_lat) .< tolerance)
        )

        if length(n_match) > 0
            n_idx = n_match[1]
            n_u_LS  = u_array[n_idx]
            n_v_LS  = v_array[n_idx]
            n_mod_LS = sqrt(n_u_LS^2 + n_v_LS^2)
        else
            n_u_LS   = NaN
            n_v_LS   = NaN
            n_mod_LS = NaN
        end

        # Store in neighbors_matrix: 12 columns per neighbor
        col = 1 + (k - 1) * 12
        neighbors_matrix[i, col + 1]  = n_u_cop
        neighbors_matrix[i, col + 2]  = n_v_cop
        neighbors_matrix[i, col + 3]  = n_mod_cop
        neighbors_matrix[i, col + 4]  = n_u_rad
        neighbors_matrix[i, col + 5]  = n_v_rad
        neighbors_matrix[i, col + 6]  = n_mod_rad
        neighbors_matrix[i, col + 7]  = n_u_tot
        neighbors_matrix[i, col + 8]  = n_v_tot
        neighbors_matrix[i, col + 9]  = n_mod_tot
        neighbors_matrix[i, col + 10] = n_u_LS
        neighbors_matrix[i, col + 11] = n_v_LS
        neighbors_matrix[i, col + 12] = n_mod_LS
    end

end

# ==================================================================================
# TEMPORAL STATISTICS CALCULATION
# ==================================================================================

println("\n" * "="^80)
println("CALCULATING TEMPORAL STATISTICS")
println("="^80)

# Filter NaNs for LS before calculating statistics
valid_LS_mask    = .!isnan.(vel_mag_LS_array)
vel_mag_LS_valid = vel_mag_LS_array[valid_LS_mask]
vel_mag_cop_valid = vel_mag_cop_array[valid_LS_mask]
n_valid_LS = sum(valid_LS_mask)

println("\nNumber of valid timesteps for LS: ", n_valid_LS, " of ", length(vel_mag_LS_array))

# 1. Time average
mean_vel_cop = mean(vel_mag_cop_array)
mean_vel_rad = mean(vel_mag_rad_array)
mean_vel_tot = mean(vel_mag_tot_array)
mean_vel_LS  = mean(vel_mag_LS_valid)

# 2. Standard deviation
std_vel_cop = std(vel_mag_cop_array)
std_vel_rad = std(vel_mag_rad_array)
std_vel_tot = std(vel_mag_tot_array)
std_vel_LS  = std(vel_mag_LS_valid)

# 3. Bias 
bias_rad = mean_vel_rad - mean_vel_cop
bias_tot = mean_vel_tot - mean_vel_cop
bias_LS  = mean_vel_LS  - mean_vel_cop

# 4. Standard deviation of reconstruction error
error_rad = vel_mag_cop_array .- vel_mag_rad_array
error_tot = vel_mag_cop_array .- vel_mag_tot_array
error_LS  = vel_mag_cop_valid .- vel_mag_LS_valid

std_error_rad = std(error_rad)
std_error_tot = std(error_tot)
std_error_LS  = std(error_LS)

# Display results
println("\nPOINT COORDINATES:")
println("  Longitude: ", closest_lon)
println("  Latitude:  ", closest_lat)

println("\nTEMPORAL AVERAGES OF VELOCITY MAGNITUDE (m/s):")
println("  Copernicus:    ", round(mean_vel_cop, digits=4))
println("  DIVAnd Radial: ", round(mean_vel_rad, digits=4))
println("  DIVAnd Total:  ", round(mean_vel_tot, digits=4))
println("  LS:            ", round(mean_vel_LS,  digits=4))

println("\nSTANDARD DEVIATION OF VELOCITY MAGNITUDE (m/s):")
println("  Copernicus:    ", round(std_vel_cop, digits=4))
println("  DIVAnd Radial: ", round(std_vel_rad, digits=4))
println("  DIVAnd Total:  ", round(std_vel_tot, digits=4))
println("  LS:            ", round(std_vel_LS,  digits=4))

println("\nMEAN BIAS (Copernicus - Method) (m/s):")
println("  DIVAnd Radial: ", round(bias_rad, digits=4))
println("  DIVAnd Total:  ", round(bias_tot, digits=4))
println("  LS:            ", round(bias_LS,  digits=4))

println("\nSTANDARD DEVIATION OF RECONSTRUCTION ERROR (m/s):")
println("  DIVAnd Radial: ", round(std_error_rad, digits=4))
println("  DIVAnd Total:  ", round(std_error_tot, digits=4))
println("  LS:            ", round(std_error_LS,  digits=4))

# Save neighbors file
output_file_neighbors = "../data/january_2026/gaps_data/stats_vel_neighbors_large_gap.txt"
# output_file_neighbors = "../data/january_2026/gaps_data/stats_vel_neighbors_small_gap.txt"

open(output_file_neighbors, "w") do io
    println(io, "# LS velocities of the 4 neighbors of the central point (lon=", closest_lon, "  lat=", closest_lat, ")")
    println(io, "# Neighbor right : lon=", neighbor_lons[1], "  lat=", neighbor_lats[1])
    println(io, "# Neighbor left  : lon=", neighbor_lons[2], "  lat=", neighbor_lats[2])
    println(io, "# Neighbor up    : lon=", neighbor_lons[3], "  lat=", neighbor_lats[3])
    println(io, "# Neighbor down  : lon=", neighbor_lons[4], "  lat=", neighbor_lats[4])
    println(io, "# Columns: timestep | u_cop_right v_cop_right mod_cop_right  u_rad_right v_rad_right mod_rad_right  u_tot_right v_tot_right mod_tot_right  u_LS_right v_LS_right mod_LS_right | same_left | same_up | same_down")
    writedlm(io, neighbors_matrix, ' ')
end

println("\nNeighbors file saved at: ", output_file_neighbors)
