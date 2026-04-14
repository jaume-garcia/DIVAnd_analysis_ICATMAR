using PyCall
using CSV, DataFrames
using DelimitedFiles
using NCDatasets

# Julia scripts
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")

# Add Python script directory to Python path
push!(PyVector(pyimport("sys")."path"), "../utils")

# Import Python scripts
hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# HFRadarProcessor instance
processor = hf.HFRadarProcessor()

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
# PROCESSING: Interpolate bathymetry to LS grid

println("Interpolating pm, pn ...")
pm_LS_grid, pn_LS_grid = ma.interp_grid_vel(xi, yi, pm, pn, lon_grid_LS, lat_grid_LS,
                                             mask_coarse_grid=[], points_mode=false)

println("Interpolating mask ...")
mask_LS_grid_raw = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

# Convert floating-point mask to boolean mask
# Assuming mask values: 1.0 = water (valid), 0.0 or NaN = land (invalid)
mask_LS_grid = mask_LS_grid_raw .>= 0.5  # Converts to Bool array

println("Interpolating h ...")
h_LS_grid = ma.interp_grid_eta(xi, yi, h, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

# ------------------------------------------------------------------------------------
# DIVAND PARAMETERS (optimized)
L = 1.5           # Correlation length scale
epsilon2 = 0.0002 # Error variance parameter
eps2_bc = 0.001   # Boundary condition parameter
eps2_divc = 1.0   # Divergence constraint parameter

# ------------------------------------------------------------------------------------
# INITIALIZE 3D ARRAYS FOR ALL TIME STEPS
n_time = 264
n_lon = size(lon_grid_LS, 1)
n_lat = size(lon_grid_LS, 2)

uri_all = zeros(Float64, n_time, n_lon, n_lat)  # u radial from DIVAnd (all times)
vri_all = zeros(Float64, n_time, n_lon, n_lat)  # v radial from DIVAnd (all times)
uti_all = zeros(Float64, n_time, n_lon, n_lat)  # u total from DIVAnd (all times)
vti_all = zeros(Float64, n_time, n_lon, n_lat)  # v total from DIVAnd (all times)

# ------------------------------------------------------------------------------------
# MAIN LOOP OVER TIME STEPS
# ------------------------------------------------------------------------------------

for i = 1:n_time
    
    println("ITERATION NUMBER = ... ", i-1)

    # TOTAL VELOCITIES (L3) FROM LS
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_"*lpad(string(i-1), 3, '0')*"_all_grid.txt"
    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = df.u_total 
    v_array = df.v_total 
    mod_vel_array = df.modulo 
    angle_array = df.angulo
    gdop_array = df.gdop

    println("LENGTH TOTAL VELOCITIES L3 WITH JAUME LS (NO RESTRICTIONS) = ... ", length(u_array))
    print("TOTALS COP read")
   
    # Filter outliers and non-physical velocities
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)
    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array = angle_array[valid_index]
    
    println("TOTALS COP read")

    # ------------------------------------------------------------------------------------
    # RADIAL VELOCITIES (FILTERED OBSERVATIONS)
    file_txt = "../data/january_2026/radials_10_days/filtered_radials/filtered_radials6_"*lpad(string(i-1), 3, '0')*".txt"
    df = CSV.read(file_txt, DataFrame,
                  header=[:obs_lon, :obs_lat, :u_radar, :v_radar, :angle_bearing,
                          :vel_radar, :angle_direction, :flag_radar],
                  skipto=1, delim=' ', ignorerepeated=true)
    
    obs_lon = df.obs_lon
    obs_lat = df.obs_lat
    u_radar = df.u_radar
    v_radar = df.v_radar
    angle_bearing = df.angle_bearing
    vel_radar = df.vel_radar
    angle_direction = df.angle_direction
    flag_radar = df.flag_radar

    lon_mesh_radar, lat_mesh_radar = meshgrid(obs_lon, obs_lat)
    
    println("RADIALS COP read")

    # ------------------------------------------------------------------------------------
    # RUN DIVAND FOR RADIALS
    uri, vri = run_divand_vel(mask_LS_grid, h_LS_grid, pm_LS_grid, pn_LS_grid,
                              lon_grid_LS, lat_grid_LS, obs_lon, obs_lat,
                              vel_radar, angle_bearing, L, epsilon2, eps2_bc, eps2_divc)
    
    # RUN DIVAND FOR TOTALS
    uti, vti = run_divand_vel(mask_LS_grid, h_LS_grid, pm_LS_grid, pn_LS_grid,
                              lon_grid_LS, lat_grid_LS, lon_array, lat_array,
                              mod_vel_array, angle_array, L, epsilon2, eps2_bc, eps2_divc)
    
    # Store results for this time step
    uri_all[i, :, :] = uri
    vri_all[i, :, :] = vri
    uti_all[i, :, :] = uti
    vti_all[i, :, :] = vti
    
    println("Iteration $i completed and saved")
end

# ------------------------------------------------------------------------------------
# SAVE RESULTS TO NETCDF FILE
# ------------------------------------------------------------------------------------

output_file = "../data/january_2026/divand/divand_field.nc"
println("Saving results to NetCDF file: $output_file")

NCDataset(output_file, "c") do ds
    
    # Define dimensions
    defDim(ds, "time", n_time)
    defDim(ds, "lon", n_lon)
    defDim(ds, "lat", n_lat)
    
    # Define coordinate variables
    nclon = defVar(ds, "longitude", Float64, ("lon", "lat"))
    nclat = defVar(ds, "latitude", Float64, ("lon", "lat"))
    nctime = defVar(ds, "time", Int32, ("time",))
    
    # Add attributes to coordinates
    nclon.attrib["long_name"] = "Longitude"
    nclon.attrib["units"] = "degrees_east"
    nclon.attrib["standard_name"] = "longitude"
    
    nclat.attrib["long_name"] = "Latitude"
    nclat.attrib["units"] = "degrees_north"
    nclat.attrib["standard_name"] = "latitude"
    
    nctime.attrib["long_name"] = "Time step"
    nctime.attrib["units"] = "time_step"
    
    # Define velocity variables
    nc_uri = defVar(ds, "u_radial_divand", Float64, ("time", "lon", "lat"))
    nc_vri = defVar(ds, "v_radial_divand", Float64, ("time", "lon", "lat"))
    nc_uti = defVar(ds, "u_total_divand", Float64, ("time", "lon", "lat"))
    nc_vti = defVar(ds, "v_total_divand", Float64, ("time", "lon", "lat"))
    
    # Add attributes to velocity variables
    nc_uri.attrib["long_name"] = "Eastward velocity from radial DIVAnd"
    nc_uri.attrib["units"] = "m/s"
    nc_uri.attrib["standard_name"] = "eastward_sea_water_velocity"
    
    nc_vri.attrib["long_name"] = "Northward velocity from radial DIVAnd"
    nc_vri.attrib["units"] = "m/s"
    nc_vri.attrib["standard_name"] = "northward_sea_water_velocity"
    
    nc_uti.attrib["long_name"] = "Eastward velocity from total DIVAnd"
    nc_uti.attrib["units"] = "m/s"
    nc_uti.attrib["standard_name"] = "eastward_sea_water_velocity"
    
    nc_vti.attrib["long_name"] = "Northward velocity from total DIVAnd"
    nc_vti.attrib["units"] = "m/s"
    nc_vti.attrib["standard_name"] = "northward_sea_water_velocity"
    
    # Write data
    nclon[:, :] = lon_grid_LS
    nclat[:, :] = lat_grid_LS
    nctime[:] = 0:(n_time-1)
    
    nc_uri[:, :, :] = uri_all
    nc_vri[:, :, :] = vri_all
    nc_uti[:, :, :] = uti_all
    nc_vti[:, :, :] = vti_all
    
    # Global attributes
    ds.attrib["title"] = "DIVAnd interpolated velocities from HF Radar"
    ds.attrib["institution"] = "ICATMAR"
    ds.attrib["source"] = "HF Radar observations"
    ds.attrib["history"] = "Created on $(now())"
    ds.attrib["DIVAnd_parameters"] = "L=$L, epsilon2=$epsilon2, eps2_bc=$eps2_bc, eps2_divc=$eps2_divc"
end

println("NetCDF file saved successfully!")
println("File location: $output_file")
