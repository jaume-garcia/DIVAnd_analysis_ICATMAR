###############################################################################
# SCIENTIFIC SCRIPT: STATISTICAL ANALYSIS OF HF RADAR VELOCITY FIELDS
#
# Description:
# This script performs a comparative analysis between:
# - Copernicus model velocities (ground truth)
# - Radial and total velocities reconstructed via DIVAnd
# - Total velocities from Least Squares (LS) Level 3 products
#
# The analysis includes:
# - Spatial interpolation
# - Grid matching
# - Computation of divergence, vorticity, and kinetic energy
# - Statistical metrics (means and correlations)
#
###############################################################################

using PyCall
using CSV, DataFrames
using DelimitedFiles
using NCDatasets

###############################################################################
# LOAD AUXILIARY JULIA SCRIPTS
###############################################################################

include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")
include("stats_module.jl")

###############################################################################
# PYTHON ENVIRONMENT CONFIGURATION
###############################################################################

# Add Python utilities directory to Python path
push!(PyVector(pyimport("sys")."path"), "../utils")

# Import Python modules
hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# Instantiate HF radar processor
processor = hf.HFRadarProcessor()

###############################################################################
# BATHYMETRY DATA LOADING
###############################################################################

file_icatmar = "../data/bathy.nc"

# Load grid and mask information
mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar)

###############################################################################
# LOAD LS GRID
###############################################################################

grid_file = "../data/hfradar_totals_grid_icatmar.nc"

lon_LS, lat_LS, lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)

# Generate meshgrid
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

###############################################################################
# LOAD COPERNICUS MODEL DATA
###############################################################################

file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"

@time lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(
    file_cop, "lon", "lat", "u_data", "v_data", "0:"
)

# Convert to meshgrid
lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:,:], lat_cop[:,:])

# Convert radians to degrees
lon_mesh_model_cop .= lon_mesh_model_cop .* (180 / pi)
lat_mesh_model_cop .= lat_mesh_model_cop .* (180 / pi)

# Replace missing values with NaN
u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)

println("Copernicus dataset successfully loaded")

###############################################################################
# GRID INTERPOLATION MASK
###############################################################################

mask_interp = ma.interp_grid_eta(
    xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[]
)

###############################################################################
# LOAD DIVAND RECONSTRUCTED FIELDS
###############################################################################

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

uri = data.u_radial
vri = data.v_radial
uti = data.u_total
vti = data.v_total

###############################################################################
# FUNCTION: MATCH LS POINTS WITH GRID AND EXTRACT DATA
###############################################################################

function map_total_points_to_LS_grid(
    lon_array, lat_array,
    lon_grid_LS, lat_grid_LS,
    u_interp_cop, v_interp_cop,
    u_array, v_array,
    uri_timestep, vri_timestep,
    uti_timestep, vti_timestep
)

    tolerance = 1e-6

    # Initialize storage vectors
    u_interp_cop_at_points = zeros(length(lon_array))
    v_interp_cop_at_points = zeros(length(lon_array))
    uri_at_points = zeros(length(lon_array))
    vri_at_points = zeros(length(lon_array))
    uti_at_points = zeros(length(lon_array))
    vti_at_points = zeros(length(lon_array))

    lon_array_at_points = zeros(length(lon_array))
    lat_array_at_points = zeros(length(lon_array))

    # Initialize 2D fields
    u_rad_2d = fill(NaN, size(lon_grid_LS))
    v_rad_2d = fill(NaN, size(lon_grid_LS))
    u_tot_2d = fill(NaN, size(lon_grid_LS))
    v_tot_2d = fill(NaN, size(lon_grid_LS))
    u_LS_2d  = fill(NaN, size(lon_grid_LS))
    v_LS_2d  = fill(NaN, size(lon_grid_LS))
    u_model_2d = fill(NaN, size(lon_grid_LS))
    v_model_2d = fill(NaN, size(lon_grid_LS))

    points_found = 0
    points_not_found = 0

    ###########################################################################
    # LOOP THROUGH ALL OBSERVATION POINTS
    ###########################################################################

    for j = 1:length(lon_array)

        # Compute distance to grid
        distances = sqrt.(
            (lon_grid_LS .- lon_array[j]).^2 .+
            (lat_grid_LS .- lat_array[j]).^2
        )

        idx_closest = argmin(distances)
        idx_lon, idx_lat = Tuple(idx_closest)

        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]

        is_grid_point =
            abs(lon_closest - lon_array[j]) < tolerance &&
            abs(lat_closest - lat_array[j]) < tolerance

        if is_grid_point
            points_found += 1
        else
            points_not_found += 1
        end

        # Assign values to grid
        u_model_2d[idx_lon, idx_lat] = u_interp_cop[idx_lon, idx_lat]
        v_model_2d[idx_lon, idx_lat] = v_interp_cop[idx_lon, idx_lat]

        u_rad_2d[idx_lon, idx_lat] = uri_timestep[idx_lon, idx_lat]
        v_rad_2d[idx_lon, idx_lat] = vri_timestep[idx_lon, idx_lat]

        u_tot_2d[idx_lon, idx_lat] = uti_timestep[idx_lon, idx_lat]
        v_tot_2d[idx_lon, idx_lat] = vti_timestep[idx_lon, idx_lat]

        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]

        # Store extracted values
        u_interp_cop_at_points[j] = u_interp_cop[idx_lon, idx_lat]
        v_interp_cop_at_points[j] = v_interp_cop[idx_lon, idx_lat]
        uri_at_points[j] = uri_timestep[idx_lon, idx_lat]
        vri_at_points[j] = vri_timestep[idx_lon, idx_lat]
        uti_at_points[j] = uti_timestep[idx_lon, idx_lat]
        vti_at_points[j] = vti_timestep[idx_lon, idx_lat]

        lon_array_at_points[j] = lon_closest
        lat_array_at_points[j] = lat_closest
    end

    ###########################################################################
    # FILTER VALID (NON-NAN) DATA
    ###########################################################################

    valid_mask = .!(
        isnan.(u_interp_cop_at_points) .|
        isnan.(v_interp_cop_at_points) .|
        isnan.(uri_at_points) .|
        isnan.(vri_at_points) .|
        isnan.(uti_at_points) .|
        isnan.(vti_at_points) .|
        isnan.(u_array) .|
        isnan.(v_array)
    )

    return u_model_2d, v_model_2d,
           u_rad_2d, v_rad_2d,
           u_tot_2d, v_tot_2d,
           u_LS_2d, v_LS_2d,
           lon_array_at_points[valid_mask],
           lat_array_at_points[valid_mask],
           u_interp_cop_at_points[valid_mask],
           v_interp_cop_at_points[valid_mask],
           uri_at_points[valid_mask],
           vri_at_points[valid_mask],
           uti_at_points[valid_mask],
           vti_at_points[valid_mask],
           u_array[valid_mask],
           v_array[valid_mask]

end

###############################################################################
# MAIN TEMPORAL LOOP
###############################################################################

n_times = 264

stats_rad = zeros(n_times, 22)

for i = 1:n_times

    println("Processing timestep: ", i)

    ###########################################################################
    # LOAD TIME-SPECIFIC DATA
    ###########################################################################

    u_cop_timestep = u_model_cop[i,:,:]
    v_cop_timestep = v_model_cop[i,:,:]

    uri_timestep = uri[i,:,:]
    vri_timestep = vri[i,:,:]
    uti_timestep = uti[i,:,:]
    vti_timestep = vti[i,:,:]

    ###########################################################################
    # LOAD LS TOTAL VELOCITIES
    ###########################################################################

    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_" *
               lpad(string(i-1), 3, '0') * "_all_grid.txt"

    df = CSV.read(file_tuv, DataFrame;
        delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = df.u_total
    v_array = df.v_total

    ###########################################################################
    # FILTER BASED ON QUALITY (GDOP AND VELOCITY MAGNITUDE)
    ###########################################################################

    valid_index = (df.gdop .<= 2.0) .& (df.speed .<= 1.2)

    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]

    ###########################################################################
    # INTERPOLATE MODEL DATA TO LS GRID
    ###########################################################################

    u_interp_cop, v_interp_cop = ma.interp_grid_vel(
        lon_mesh_model_cop, lat_mesh_model_cop,
        u_cop_timestep, v_cop_timestep,
        lon_grid_LS, lat_grid_LS,
        mask_coarse_grid=[], points_mode=false
    )

    ###########################################################################
    # MAP DATA TO GRID
    ###########################################################################

    u_model_2d, v_model_2d,
    u_rad_2d, v_rad_2d,
    u_tot_2d, v_tot_2d,
    u_LS_2d, v_LS_2d,
    lon_array_filtered, lat_array_filtered,
    u_interp_cop_filtered, v_interp_cop_filtered,
    uri_filtered, vri_filtered,
    uti_filtered, vti_filtered,
    u_array_filtered, v_array_filtered =
        map_total_points_to_LS_grid(
            lon_array, lat_array,
            lon_grid_LS, lat_grid_LS,
            u_interp_cop, v_interp_cop,
            u_array, v_array,
            uri_timestep, vri_timestep,
            uti_timestep, vti_timestep
        )

    ###########################################################################
    # COMPUTE PHYSICAL QUANTITIES
    ###########################################################################

    vorticity_model = calculate_vorticity(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    divergence_model = calculate_divergence(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)

    ke_model = calculate_kinetic_energy(u_model_2d, v_model_2d)

    ###########################################################################
    # STORE STATISTICS (SIMPLIFIED FOR BREVITY)
    ###########################################################################

    stats_rad[i, 1] = i - 1
    stats_rad[i, 2] = mean(skipmissing(divergence_model))
    stats_rad[i, 6] = mean(skipmissing(vorticity_model))
    stats_rad[i,10] = mean(skipmissing(ke_model))

end

###############################################################################
# OUTPUT RESULTS
###############################################################################

output_file = "../data/january_2026/stats_data/all_stats_phys_grid_icatmar.txt"
writedlm(output_file, stats_rad, ' ')

println("ASCII file generated: ", output_file)

###############################################################################
# GLOBAL CORRELATION STATISTICS
###############################################################################

using Statistics
using Printf

println("\n=================================================================")
println("MEAN CORRELATION COEFFICIENTS")
println("=================================================================")
