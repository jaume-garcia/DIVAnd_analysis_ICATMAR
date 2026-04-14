using DIVAnd
using DIVAnd_HFRadar: DIVAndrun_HFRadar
using Statistics
using NCDatasets
using MeshGrid
using Interpolations

# -----------------------------------------------------------------------
# Module: divand_process.jl
# Purpose: Julia functions for reconstructing 2D total velocity fields
#          from HF radar radial observations using the variational
#          interpolation method DIVAnd (Data-Interpolating Variational
#          Analysis in n dimensions).
# -----------------------------------------------------------------------


"""
    create_grid(dx, dy, lon_min, lon_max, lat_min, lat_max)

Creates a regular rectangular grid for the DIVAnd simulation domain.

# Arguments
- `dx`      : Longitude grid spacing (degrees)
- `dy`      : Latitude grid spacing (degrees)
- `lon_min` : Western boundary of the domain (degrees)
- `lon_max` : Eastern boundary of the domain (degrees)
- `lat_min` : Southern boundary of the domain (degrees)
- `lat_max` : Northern boundary of the domain (degrees)

# Returns
- `lonr`          : Longitude range (StepRange)
- `latr`          : Latitude range (StepRange)
- `mask`          : Boolean ocean/land mask
- `(pm, pn)`      : Metric coefficients (reciprocal grid spacings in x and y)
- `(xi, yi)`      : 2D longitude and latitude grids
"""
function create_grid(dx, dy, lon_min, lon_max, lat_min, lat_max)

    lonr = lon_min:dx:lon_max
    latr = lat_min:dy:lat_max

    # Build the DIVAnd rectangular domain (mask, metric coefficients, coordinate grids)
    mask, (pm, pn), (xi, yi) = DIVAnd_rectdom(lonr, latr)

    return lonr, latr, mask, pm, pn, xi, yi
end


"""
    create_mask(file_bath, lonr, latr)

Creates a Boolean ocean/land mask from a bathymetry file.
Downloads the bathymetry file if it is not already present locally.

# Arguments
- `file_bath` : Local path to the bathymetry file (downloaded if absent)
- `lonr`      : Longitude range of the target grid
- `latr`      : Latitude range of the target grid

# Returns
- `mask` : Boolean matrix (true = ocean, false = land)
- `b`    : Bathymetric depth matrix (positive downward)
"""
function create_mask(file_bath, lonr, latr)

    # Download the bathymetry file if it does not exist locally
    if !isfile(file_bath)
        download("https://dox.uliege.be/index.php/s/RSwm4HPHImdZoQP/download", file_bath)
    else
        @info("Bathymetry file already downloaded")
    end

    # Load the bathymetry interpolated to the target grid
    bx, by, b = load_bath(file_bath, true, lonr, latr)

    # Initialise mask as all false (land)
    mask = falses(size(b, 1), size(b, 2))

    # Mark grid points with depth >= 1 m as ocean
    for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            mask[i, j] = b[i, j] >= 1.0
        end
    end

    return mask, b
end


"""
    run_divand_vel(mask, b, pm, pn, xi, yi, obs_lon, obs_lat, r_vel, angle,
                  len, epsilon2, eps2_bc, eps2_divc)

Runs the DIVAnd variational analysis to reconstruct total (u, v) velocity
components from HF radar radial velocity observations.

# Arguments
- `mask`        : Boolean ocean/land mask (BitMatrix)
- `b`           : Bathymetric depth matrix
- `pm, pn`      : Metric coefficients (reciprocal grid spacings)
- `xi, yi`      : 2D longitude and latitude grids
- `obs_lon`     : Longitudes of the radial observations
- `obs_lat`     : Latitudes of the radial observations
- `r_vel`       : Radial velocity observations (m/s)
- `angle`       : Bearing angles of the radial observations (degrees)
- `len`         : Correlation length scale(s) for the analysis
- `epsilon2`    : Observation error-to-signal variance ratio
- `eps2_bc`     : Weight for the boundary condition constraint
- `eps2_divc`   : Weight for the divergence constraint

# Returns
- `uri, vri` : Reconstructed east and north velocity fields on the grid
"""
function run_divand_vel(mask, b, pm, pn, xi, yi, obs_lon, obs_lat, r_vel, angle,
                        len, epsilon2, eps2_bc, eps2_divc)

    # Convert the mask to BitMatrix as required by DIVAndrun_HFRadar
    mask_bit = convert(BitMatrix, mask)

    # Run the HF radar DIVAnd analysis
    uri, vri = DIVAndrun_HFRadar(
        mask_bit, b, (pm, pn), (xi, yi), (obs_lon, obs_lat),
        r_vel, angle, len, epsilon2,
        eps2_boundary_constraint = eps2_bc,
        eps2_div_constraint      = eps2_divc,
    )

    return uri, vri
end


"""
    grid_icatmar(file_icatmar)

Reads the ICATMAR (Institut Català de Recerca per a la Governança del Mar)
regular grid from a NetCDF file and returns the grid arrays required by DIVAnd.

# Arguments
- `file_icatmar` : Path to the ICATMAR NetCDF grid file

# Returns
- `mask`     : Boolean ocean/land mask (true = ocean)
- `h_clean`  : Bathymetric depth array (NaN at land points)
- `pm, pn`   : Metric coefficients (reciprocal grid spacings in x and y)
- `lon_grid` : 2D longitude grid
- `lat_grid` : 2D latitude grid
"""
function grid_icatmar(file_icatmar)

    ds = NCDataset(file_icatmar, "r")

    # Read longitude, latitude, and elevation variables
    lon_ds = ds["lon"]
    lat_ds = ds["lat"]
    h_ds   = ds["elevation"]

    lon_ds = lon_ds[:]
    lat_ds = lat_ds[:]
    h      = h_ds[:, :]

    # Replace missing values with NaN32 for numerical processing
    h_clean   = replace(h,      missing => NaN32)
    lon_clean = replace(lon_ds, missing => NaN32)
    lat_clean = replace(lat_ds, missing => NaN32)

    # Close the NetCDF file
    close(ds)

    # Build the ocean mask: grid points with negative elevation are ocean
    mask = BitMatrix(h_clean .< 0)

    # Compute metric coefficients (reciprocal of grid spacing in degrees)
    sz = (size(lon_clean)[1], size(lat_clean)[1])
    pm = ones(sz) / (lon_clean[2] - lon_clean[1])   # 1/Δlon
    pn = ones(sz) / (lat_clean[2] - lat_clean[1])   # 1/Δlat

    # Build the 2D coordinate grids using meshgrid
    lat_grid, lon_grid = meshgrid(lat_clean, lon_clean)

    return mask, h_clean, pm, pn, lon_grid, lat_grid
end


"""
    rms_mb(ui, vi, u_obs, v_obs)

Computes error statistics comparing the reconstructed velocity field against
point observations: Mean Square Error (MSE), Root Mean Square Error (RMSE),
and Mean Bias (MB) for both components and the total velocity.

# Arguments
- `ui`    : Reconstructed east velocity at observation locations
- `vi`    : Reconstructed north velocity at observation locations
- `u_obs` : Observed east velocity
- `v_obs` : Observed north velocity

# Returns
- `u_mse, v_mse`       : MSE for the u and v components
- `total_mse`          : Total MSE (combined u and v)
- `u_rmse, v_rmse`     : RMSE for the u and v components
- `total_rmse`         : Total RMSE
- `u_mb, v_mb`         : Mean bias for the u and v components
- `total_mb`           : Total mean bias
"""
function rms_mb(ui, vi, u_obs, v_obs)

    println("U_OBS = ... ", u_obs)

    # Compute residuals
    u_diff = ui .- u_obs
    v_diff = vi .- v_obs

    # Mean Square Error for each component
    u_mse     = mean(u_diff .^ 2)
    v_mse     = mean(v_diff .^ 2)
    total_mse = mean(u_diff .^ 2 .+ v_diff .^ 2)

    # Root Mean Square Error
    u_rmse     = sqrt(u_mse)
    v_rmse     = sqrt(v_mse)
    total_rmse = sqrt(total_mse)

    # Mean Bias (mean of the residuals)
    u_mb     = mean(u_diff)
    v_mb     = mean(v_diff)
    total_mb = mean(u_diff + v_diff)

    return u_mse, v_mse, total_mse, u_rmse, v_rmse, total_rmse, u_mb, v_mb, total_mb
end


"""
    interpolation_point(xi, yi, uri, vri, obs_lon, obs_lat)

Interpolates the reconstructed velocity field (uri, vri) defined on a regular
grid (xi, yi) to a set of scattered observation points (obs_lon, obs_lat)
using linear interpolation.

# Arguments
- `xi, yi`           : Unique sorted longitude and latitude vectors of the grid
- `uri, vri`         : Reconstructed east and north velocity fields (1D or 2D)
- `obs_lon, obs_lat` : Longitudes and latitudes of the target observation points

# Returns
- `interpolated_u` : Interpolated east velocity at each observation point
- `interpolated_v` : Interpolated north velocity at each observation point
"""
function interpolation_point(xi, yi, uri, vri, obs_lon, obs_lat)

    # Ensure grid coordinates are unique and sorted
    xi_sorted = sort(unique(xi))
    yi_sorted = sort(unique(yi))

    # Reshape velocity fields to 2D matrices if necessary
    uri_matrix = reshape(uri, length(xi_sorted), length(yi_sorted))
    vri_matrix = reshape(vri, length(xi_sorted), length(yi_sorted))

    # Build bilinear interpolators on the regular grid
    itp_uri = interpolate((xi_sorted, yi_sorted), uri_matrix, Gridded(Linear()))
    itp_vri = interpolate((xi_sorted, yi_sorted), vri_matrix, Gridded(Linear()))

    # Evaluate the interpolators at each observation point
    interpolated_u = [itp_uri(lon, lat) for (lon, lat) in zip(obs_lon, obs_lat)]
    interpolated_v = [itp_vri(lon, lat) for (lon, lat) in zip(obs_lon, obs_lat)]

    return interpolated_u, interpolated_v
end


"""
    read_divand_file(filename)

Reads a DIVAnd output NetCDF file and returns the stored velocity fields
and grid coordinates.

# Arguments
- `filename` : Path to the DIVAnd NetCDF output file

# Returns
- `lon`  : 2D longitude grid
- `lat`  : 2D latitude grid
- `time` : Time vector
- `uri`  : Radially reconstructed east velocity field (lon × lat × time)
- `vri`  : Radially reconstructed north velocity field (lon × lat × time)
- `uti`  : Total east velocity field from DIVAnd (lon × lat × time)
- `vti`  : Total north velocity field from DIVAnd (lon × lat × time)
"""
function read_divand_file(filename)

    ds  = Dataset(filename)
    lon = ds["longitude"][:, :]
    lat = ds["latitude"][:, :]
    time = ds["time"][:]

    # Radial velocity components from DIVAnd
    uri = ds["u_radial_divand"][:, :, :]
    vri = ds["v_radial_divand"][:, :, :]

    # Total velocity components from DIVAnd
    uti = ds["u_total_divand"][:, :, :]
    vti = ds["v_total_divand"][:, :, :]

    close(ds)

    return lon, lat, time, uri, vri, uti, vti
end

