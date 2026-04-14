using CSV
using DataFrames
using NCDatasets
using MeshGrid
using Dates

# -----------------------------------------------------------------------
# Module: reading_obs.jl
# Purpose: Julia I/O utilities for reading HF radar observations, drifting
#          buoy data, and total velocity fields from CSV, TUV, and NetCDF
#          files used in the Catalan Sea DIVAnd reconstruction project.
# -----------------------------------------------------------------------


"""
    extract_date(file) -> DateTime or nothing

Extracts a DateTime object from an HF radar filename by matching the
pattern YYYY_MM_DD_HHMM.

# Arguments
- `file` : Filename string (e.g. "RDL_AREN_2022_01_15_1200.ruv")

# Returns
- `DateTime` object if the pattern is found; `nothing` otherwise
"""
function extract_date(file)
    m = match(r"(\d{4})_(\d{2})_(\d{2})_(\d{4})", file)
    return m !== nothing ? DateTime(parse(Int, m[1]), parse(Int, m[2]),
                                    parse(Int, m[3]), parse(Int, m[4]) ÷ 100, 0) : nothing
end


"""
    read_obs_csv(file_csv) -> (obs_lon, obs_lat, u_rad, v_rad,
                               angle_bearing, r_vel, angle_direction, flag_radar)

Reads a semicolon-separated CSV radial observation file and returns the
relevant oceanographic variables.

Expected column order (no header):
  1: longitude, 2: latitude, 3: u (cm/s), 4: v (cm/s), 5: bearing angle,
  6: radial velocity (cm/s), 7: direction angle, 8: antenna flag

# Arguments
- `file_csv` : Path to the CSV observation file

# Returns
- `obs_lon`         : Longitude of each observation (degrees)
- `obs_lat`         : Latitude of each observation (degrees)
- `u_rad`           : East velocity component (m/s; converted from cm/s)
- `v_rad`           : North velocity component (m/s; converted from cm/s)
- `angle_bearing`   : Bearing angle (degrees)
- `r_vel`           : Radial velocity magnitude (m/s; converted from cm/s)
- `angle_direction` : Direction angle (degrees)
- `flag_radar`      : Antenna/radar flag identifier
"""
function read_obs_csv(file_csv)

    # Read the CSV file into a DataFrame and convert to a plain matrix
    data_vel_csv = CSV.read(file_csv, DataFrame; delim=';', header=false)
    matrix_data  = Matrix(data_vel_csv)

    # Extract observation arrays; convert velocities from cm/s to m/s
    obs_lon         = matrix_data[:, 1]
    obs_lat         = matrix_data[:, 2]
    u_rad           = matrix_data[:, 3]
    v_rad           = matrix_data[:, 4]
    angle_bearing   = matrix_data[:, 5]
    r_vel           = matrix_data[:, 6]
    angle_direction = matrix_data[:, 7]
    flag_radar      = matrix_data[:, 8]

    return obs_lon, obs_lat, u_rad, v_rad, angle_bearing, r_vel, angle_direction, flag_radar
end


"""
    read_dummy_csv(file_csv) -> (obs_time, obs_lon, obs_lat, r_vel, angle)

Reads a dummy/synthetic radial observation CSV file and returns the
relevant variables (no time filtering).

# Arguments
- `file_csv` : Path to the dummy CSV file

# Returns
- `obs_time` : Observation timestamps (column 2)
- `obs_lon`  : Longitudes (degrees)
- `obs_lat`  : Latitudes (degrees)
- `r_vel`    : Radial velocity magnitudes
- `angle`    : Bearing angles (degrees)
"""
function read_dummy_csv(file_csv)

    # Read the dummy CSV file (note: the original code used file_dummy_csv internally)
    data_vel_csv = CSV.read(file_dummy_csv, DataFrame; delim=';', header=false)
    matrix_data  = Matrix(data_vel_csv)

    obs_time = matrix_data[:, 2]
    obs_lat  = matrix_data[:, 3]
    obs_lon  = matrix_data[:, 4]
    angle    = matrix_data[:, 6]
    r_vel    = matrix_data[:, 7]

    return obs_time, obs_lon, obs_lat, r_vel, angle
end


"""
    read_dummy_csv(file_csv, time_min, time_max) -> (obs_time_filtered,
        obs_lon_filtered, obs_lat_filtered, r_vel_filtered, angle_filtered)

Overloaded version of `read_dummy_csv` that filters the data to a
temporal range defined by `time_min` and `time_max`.

# Arguments
- `file_csv` : Path to the dummy CSV file
- `time_min` : Start of the time range as "dd/mm/yyyy HH:MM:SS"
- `time_max` : End of the time range as "dd/mm/yyyy HH:MM:SS"

# Returns
- Filtered versions of obs_time, obs_lon, obs_lat, r_vel, and angle
"""
function read_dummy_csv(file_csv, time_min::String, time_max::String)

    # Read the dummy CSV file
    data_vel_csv = CSV.read(file_dummy_csv, DataFrame; delim=';', header=false)
    matrix_data  = Matrix(data_vel_csv)

    raw_obs_time = matrix_data[:, 2]
    obs_lat      = matrix_data[:, 3]
    obs_lon      = matrix_data[:, 4]
    angle        = matrix_data[:, 6]
    r_vel        = matrix_data[:, 7]

    # Parse raw time strings into DateTime objects
    obs_time = DateTime.(raw_obs_time, "dd/mm/yyyy HH:MM:SS")

    # Define the time range boundaries
    t_min = DateTime(time_min, "dd/mm/yyyy HH:MM:SS")
    t_max = DateTime(time_max, "dd/mm/yyyy HH:MM:SS")

    # Build a boolean mask for observations within the time range
    mask = (obs_time .>= t_min) .& (obs_time .<= t_max)

    # Apply the time filter
    obs_time_filtered = obs_time[mask]
    obs_lat_filtered  = obs_lat[mask]
    obs_lon_filtered  = obs_lon[mask]
    angle_filtered    = angle[mask]
    r_vel_filtered    = r_vel[mask]

    return obs_time_filtered, obs_lon_filtered, obs_lat_filtered, r_vel_filtered, angle_filtered
end


"""
    read_obs_total_nc(file_vel_nc) -> (obs_lon_grid, obs_lat_grid, U_clean, V_clean)

Reads a total velocity NetCDF file (L3 product) and returns the cleaned
velocity field as 2D arrays, with unrealistically high velocities set to zero.

# Arguments
- `file_vel_nc` : Path to the total velocity NetCDF file

# Returns
- `obs_lon_grid` : 2D longitude grid
- `obs_lat_grid` : 2D latitude grid
- `U_clean`      : East velocity component (NaN for missing; zeroed if |v| > 1.5 m/s)
- `V_clean`      : North velocity component (NaN for missing; zeroed if |v| > 1.5 m/s)
"""
function read_obs_total_nc(file_vel_nc)

    ds = NCDataset(file_vel_nc, "r")

    # Extract longitude, latitude, and velocity variables
    lon_ds = ds["longitude"]
    lat_ds = ds["latitude"]
    u_ds   = ds["u"]
    v_ds   = ds["v"]

    obs_lon_ds = lon_ds[:]
    obs_lat_ds = lat_ds[:]
    obs_u_ds   = u_ds[:, :, 1, 1]   # First time and depth slice
    obs_v_ds   = v_ds[:, :, 1, 1]

    # Build 2D coordinate meshgrids
    obs_lat_grid, obs_lon_grid = meshgrid(obs_lat_ds, obs_lon_ds)

    # Replace missing values with NaN32
    U_clean = replace(obs_u_ds, missing => NaN32)
    V_clean = replace(obs_v_ds, missing => NaN32)

    # Set to zero any velocity vectors with magnitude > 1.5 m/s (outlier filter)
    module_vel = sqrt.(U_clean .^ 2 .+ V_clean .^ 2)
    U_clean[module_vel .> 1.5] .= 0
    V_clean[module_vel .> 1.5] .= 0

    return obs_lon_grid, obs_lat_grid, U_clean, V_clean
end


"""
    clean_obs_total(lon, lat, u, v, mod_vel, angle) -> (lon_array, lat_array,
        u_array, v_array, mod_vel_array, angle_array)

Parses string-format total velocity observation data, converts units from
cm/s to m/s, and removes physically unrealistic velocities (|v| > 1 m/s).

# Arguments
- `lon`, `lat`     : Longitude and latitude (as strings)
- `u`, `v`         : East and north velocity components (cm/s, as strings)
- `mod_vel`        : Velocity magnitude (cm/s, as strings)
- `angle`          : Direction angle (degrees, as strings)

# Returns
- Filtered float arrays for lon, lat, u, v, mod_vel, and angle
"""
function clean_obs_total(lon, lat, u, v, mod_vel, angle)

    # Parse string arrays to Float64
    lon_array     = parse.(Float64, lon)
    lat_array     = parse.(Float64, lat)
    u_array       = parse.(Float64, u)       ./ 100   # cm/s → m/s
    v_array       = parse.(Float64, v)       ./ 100   # cm/s → m/s
    mod_vel_array = parse.(Float64, mod_vel) ./ 100   # cm/s → m/s
    angle_array   = parse.(Float64, angle)

    # Remove outliers: keep only observations with magnitude <= 1 m/s
    valid_index = mod_vel_array .<= 1.

    lon_array     = lon_array[valid_index]
    lat_array     = lat_array[valid_index]
    u_array       = u_array[valid_index]
    v_array       = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array   = angle_array[valid_index]

    return lon_array, lat_array, u_array, v_array, mod_vel_array, angle_array
end


"""
    read_buoys(filtered_date=""; file_tarr=nothing, file_begu=nothing,
               file_maho=nothing, file_drag=nothing, file_vale=nothing)
    -> (buoy_names, time, lon, lat, mod_vel, angle, u, v)

Reads surface current data from up to five Puertos del Estado directional
buoys (Tarragona, Begur, Mahón, Dragonera, Valencia). Each buoy file is
a tab-separated text file with a 2-row header.

If `filtered_date` is non-empty, only the record matching that date string
is returned. Otherwise, all valid records are returned.

# Arguments
- `filtered_date` : Date string to filter by (empty string = return all records)
- `file_tarr`     : Path to the Tarragona buoy file (optional)
- `file_begu`     : Path to the Begur buoy file (optional)
- `file_maho`     : Path to the Mahón buoy file (optional)
- `file_drag`     : Path to the Dragonera buoy file (optional)
- `file_vale`     : Path to the Valencia buoy file (optional)

# Returns
- `buoy_names` : Vector of buoy name strings for each returned record
- `time`       : Time vector from the last buoy file read
- `lon`, `lat` : Longitude and latitude vectors (degrees)
- `mod_vel`    : Velocity magnitude vector (m/s)
- `angle`      : Current direction vector (degrees)
- `u`, `v`     : East and north velocity component vectors (m/s)
"""
function read_buoys(filtered_date=""; file_tarr=nothing, file_begu=nothing,
                    file_maho=nothing, file_drag=nothing, file_vale=nothing)

    # Initialise output vectors
    buoy_names = String[]
    lon        = Float64[]
    lat        = Float64[]
    mod_vel    = Float64[]
    angle      = Float64[]
    u          = Float64[]
    v          = Float64[]

    # ── TARRAGONA BUOY ──────────────────────────────────────────────────
    if file_tarr !== nothing
        lon_tarr = 1.47
        lat_tarr = 40.69

        df   = CSV.File(file_tarr; delim='\t', skipto=3) |> DataFrame
        time = df[:, 1]
        vel  = df[:, 2]
        ang  = df[:, 3]

        if filtered_date != ""
            # Return only the record matching the requested date
            index_filtered = findall(time .== filtered_date)

            if !isempty(index_filtered)
                mod_vel_tarr = vel[index_filtered][1] / 100   # cm/s → m/s
                angle_tarr   = ang[index_filtered][1]
                u_tarr       = mod_vel_tarr * sin(angle_tarr * pi / 180)
                v_tarr       = mod_vel_tarr * cos(angle_tarr * pi / 180)

                push!(buoy_names, "Tarragona")
                push!(lon, lon_tarr)
                push!(lat, lat_tarr)
                push!(mod_vel, mod_vel_tarr)
                push!(angle, angle_tarr)
                push!(u, u_tarr)
                push!(v, v_tarr)
            end
        else
            # Return all valid records
            for i in 1:length(time)
                if vel[i] != "" && ang[i] != ""   # Ensure valid data exists
                    mod_vel_tarr = vel[i] / 100    # cm/s → m/s
                    angle_tarr   = ang[i]
                    u_tarr       = mod_vel_tarr * sin(angle_tarr * pi / 180)
                    v_tarr       = mod_vel_tarr * cos(angle_tarr * pi / 180)

                    push!(buoy_names, "Tarragona")
                    push!(lon, lon_tarr)
                    push!(lat, lat_tarr)
                    push!(mod_vel, mod_vel_tarr)
                    push!(angle, angle_tarr)
                    push!(u, u_tarr)
                    push!(v, v_tarr)
                end
            end
        end
    end

    # ── BEGUR BUOY ──────────────────────────────────────────────────────
    if file_begu !== nothing
        lon_begu = 3.65
        lat_begu = 41.90

        df   = CSV.File(file_begu; delim='\t', skipto=3) |> DataFrame
        time = df[:, 1]
        vel  = df[:, 2]
        ang  = df[:, 3]

        if filtered_date != ""
            index_filtered = findall(time .== filtered_date)

            if !isempty(index_filtered)
                mod_vel_begu = vel[index_filtered][1] / 100
                angle_begu   = ang[index_filtered][1]
                u_begu       = mod_vel_begu * sin(angle_begu * pi / 180)
                v_begu       = mod_vel_begu * cos(angle_begu * pi / 180)

                push!(buoy_names, "Begur")
                push!(lon, lon_begu)
                push!(lat, lat_begu)
                push!(mod_vel, mod_vel_begu)
                push!(angle, angle_begu)
                push!(u, u_begu)
                push!(v, v_begu)
            end
        else
            for i in 1:length(time)
                if vel[i] != "" && ang[i] != ""
                    mod_vel_begu = vel[i] / 100
                    angle_begu   = ang[i]
                    u_begu       = mod_vel_begu * sin(angle_begu * pi / 180)
                    v_begu       = mod_vel_begu * cos(angle_begu * pi / 180)

                    push!(buoy_names, "Begur")
                    push!(lon, lon_begu)
                    push!(lat, lat_begu)
                    push!(mod_vel, mod_vel_begu)
                    push!(angle, angle_begu)
                    push!(u, u_begu)
                    push!(v, v_begu)
                end
            end
        end
    end

    # ── MAHÓN BUOY ──────────────────────────────────────────────────────
    if file_maho !== nothing
        lon_maho = 4.42
        lat_maho = 39.71

        df   = CSV.File(file_maho; delim='\t', skipto=3) |> DataFrame
        time = df[:, 1]
        vel  = df[:, 2]
        ang  = df[:, 3]

        if filtered_date != ""
            index_filtered = findall(time .== filtered_date)

            if !isempty(index_filtered)
                mod_vel_maho = vel[index_filtered][1] / 100
                angle_maho   = ang[index_filtered][1]
                u_maho       = mod_vel_maho * sin(angle_maho * pi / 180)
                v_maho       = mod_vel_maho * cos(angle_maho * pi / 180)

                push!(buoy_names, "Mahon")
                push!(lon, lon_maho)
                push!(lat, lat_maho)
                push!(mod_vel, mod_vel_maho)
                push!(angle, angle_maho)
                push!(u, u_maho)
                push!(v, v_maho)
            end
        else
            for i in 1:length(time)
                if vel[i] != "" && ang[i] != ""
                    mod_vel_maho = vel[i] / 100
                    angle_maho   = ang[i]
                    u_maho       = mod_vel_maho * sin(angle_maho * pi / 180)
                    v_maho       = mod_vel_maho * cos(angle_maho * pi / 180)

                    push!(buoy_names, "Mahon")
                    push!(lon, lon_maho)
                    push!(lat, lat_maho)
                    push!(mod_vel, mod_vel_maho)
                    push!(angle, angle_maho)
                    push!(u, u_maho)
                    push!(v, v_maho)
                end
            end
        end
    end

    # ── DRAGONERA BUOY ──────────────────────────────────────────────────
    if file_drag !== nothing
        lon_drag = 2.10
        lat_drag = 39.56

        df   = CSV.File(file_drag; delim='\t', skipto=3) |> DataFrame
        time = df[:, 1]
        vel  = df[:, 2]
        ang  = df[:, 3]

        if filtered_date != ""
            index_filtered = findall(time .== filtered_date)

            if !isempty(index_filtered)
                mod_vel_drag = vel[index_filtered][1] / 100
                angle_drag   = ang[index_filtered][1]
                u_drag       = mod_vel_drag * sin(angle_drag * pi / 180)
                v_drag       = mod_vel_drag * cos(angle_drag * pi / 180)

                push!(buoy_names, "Dragonera")
                push!(lon, lon_drag)
                push!(lat, lat_drag)
                push!(mod_vel, mod_vel_drag)
                push!(angle, angle_drag)
                push!(u, u_drag)
                push!(v, v_drag)
            end
        else
            for i in 1:length(time)
                if vel[i] != "" && ang[i] != ""
                    mod_vel_drag = vel[i] / 100
                    angle_drag   = ang[i]
                    u_drag       = mod_vel_drag * sin(angle_drag * pi / 180)
                    v_drag       = mod_vel_drag * cos(angle_drag * pi / 180)

                    push!(buoy_names, "Dragonera")
                    push!(lon, lon_drag)
                    push!(lat, lat_drag)
                    push!(mod_vel, mod_vel_drag)
                    push!(angle, angle_drag)
                    push!(u, u_drag)
                    push!(v, v_drag)
                end
            end
        end
    end

    # ── VALENCIA BUOY ───────────────────────────────────────────────────
    if file_vale !== nothing
        lon_vale = 0.20
        lat_vale = 39.51

        df   = CSV.File(file_vale; delim='\t', skipto=3) |> DataFrame
        time = df[:, 1]
        vel  = df[:, 2]
        ang  = df[:, 3]

        if filtered_date != ""
            index_filtered = findall(time .== filtered_date)

            if !isempty(index_filtered)
                mod_vel_vale = vel[index_filtered][1] / 100
                angle_vale   = ang[index_filtered][1]
                u_vale       = mod_vel_vale * sin(angle_vale * pi / 180)
                v_vale       = mod_vel_vale * cos(angle_vale * pi / 180)

                push!(buoy_names, "Valencia")
                push!(lon, lon_vale)
                push!(lat, lat_vale)
                push!(mod_vel, mod_vel_vale)
                push!(angle, angle_vale)
                push!(u, u_vale)
                push!(v, v_vale)
            end
        else
            for i in 1:length(time)
                if vel[i] != "" && ang[i] != ""
                    mod_vel_vale = vel[i] / 100
                    angle_vale   = ang[i]
                    u_vale       = mod_vel_vale * sin(angle_vale * pi / 180)
                    v_vale       = mod_vel_vale * cos(angle_vale * pi / 180)

                    push!(buoy_names, "Valencia")
                    push!(lon, lon_vale)
                    push!(lat, lat_vale)
                    push!(mod_vel, mod_vel_vale)
                    push!(angle, angle_vale)
                    push!(u, u_vale)
                    push!(v, v_vale)
                end
            end
        end
    end

    return buoy_names, time, lon, lat, mod_vel, angle, u, v
end


"""
    read_codar_nc(file_codar) -> (lon_clean_codar, lat_clean_codar,
        u_clean_codar, v_clean_codar, r_vel_clean_codar, angle_clean_codar)

Reads a Puertos del Estado CODAR radar total velocity NetCDF file, computes
the radial velocity magnitude and direction angle, and returns only the
grid points with valid (non-NaN) data.

# Arguments
- `file_codar` : Path to the CODAR NetCDF file

# Returns
- `lon_clean_codar`   : Longitudes of valid grid points
- `lat_clean_codar`   : Latitudes of valid grid points
- `u_clean_codar`     : East velocity component at valid points (m/s)
- `v_clean_codar`     : North velocity component at valid points (m/s)
- `r_vel_clean_codar` : Velocity magnitude at valid points (m/s)
- `angle_clean_codar` : Current direction at valid points (degrees)
"""
function read_codar_nc(file_codar)

    ds = NCDataset(file_codar, "r")

    # Extract longitude, latitude, and velocity variables
    lon_ds = ds["lon"]
    lat_ds = ds["lat"]
    u_ds   = ds["u"]
    v_ds   = ds["v"]

    obs_lon_codar = lon_ds[:]
    obs_lat_codar = lat_ds[:]
    obs_u_codar   = u_ds[:, :, 1]   # First time slice
    obs_v_codar   = v_ds[:, :, 1]

    # Replace missing values with NaN32
    u_clean_codar = replace(obs_u_codar, missing => NaN32)
    v_clean_codar = replace(obs_v_codar, missing => NaN32)

    # Build 2D coordinate meshgrids
    obs_lat_codar_grid, obs_lon_codar_grid = meshgrid(obs_lat_codar, obs_lon_codar)

    # Compute velocity magnitude and direction angle (degrees)
    r_vel_codar   = sqrt.(u_clean_codar .^ 2 .+ u_clean_codar .^ 2)  # Note: uses u twice (original code)
    theta_codar   = atan.(v_clean_codar, u_clean_codar)               # radians
    theta_deg_codar = theta_codar * 180 / pi                          # degrees

    # Retain only grid points with a valid (non-NaN) velocity magnitude
    valid_indices = findall(!isnan, r_vel_codar)

    u_clean_codar       = u_clean_codar[valid_indices]
    v_clean_codar       = v_clean_codar[valid_indices]
    r_vel_clean_codar   = r_vel_codar[valid_indices]
    angle_clean_codar   = theta_deg_codar[valid_indices]
    lat_clean_codar     = obs_lat_codar_grid[valid_indices]
    lon_clean_codar     = obs_lon_codar_grid[valid_indices]

    close(ds)

    return lon_clean_codar, lat_clean_codar, u_clean_codar, v_clean_codar,
           r_vel_clean_codar, angle_clean_codar
end
