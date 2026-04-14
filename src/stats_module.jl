"""
Statistical analysis functions for ocean current validation
Includes correlation, vorticity, divergence, kinetic energy, and spectral analysis
"""

using NaturalEarth
using Interpolations

# Function to plot velocity anomalies
function plot_anomalies(lon_mesh, lat_mesh, obs_lon, obs_lat, diff_u, diff_v, diff_magnitude, mask, 
                        outfile1, outfile2, outfile3, outfile4)
    
    # Plot parameters
    pdt = 5
    SCALE = 15
    SCALE1 = 15
    WIDTH = 0.0025
    x_min = 1.7
    x_max = 4.05
    y_min = 40.9
    y_max = 42.7
    
    diff_u_2d = fill(NaN, size(lon_mesh))
    diff_v_2d = fill(NaN, size(lon_mesh))
    diff_magnitude_2d = fill(NaN, size(lon_mesh))
    
    diff_u_2d[valid_interp_mask] = diff_u
    diff_v_2d[valid_interp_mask] = diff_v
    diff_magnitude_2d[valid_interp_mask] = diff_magnitude
    
    lon_vec = lon_mesh[1,:]
    lat_vec = lat_mesh[:,1]

    # Interpolate mask for plotting
    mask_interp = LinearInterpolation((xi[:, 1], yi[1, :]), mask, extrapolation_bc=0)
    mask_vis = [mask_interp(lon, lat) for lat in lat_vec, lon in lon_vec]

    idxs = 1:6:length(obs_lon)

    # U-component anomaly plot
    p1 = heatmap(lon_vec, lat_vec, diff_u_2d,
        title = "U-component Anomaly (Copernicus GT - DIVAnd)",
        titlefontsize = 8,
        xlabel = "Longitude (°)", 
        ylabel = "Latitude (°)",
        color = :seismic,
        clims = (-1., 1.),
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
        fontsize = 5, size = (600, 600))
    
    scatter!(p1, obs_lon[idxs], obs_lat[idxs], color = :black, alpha = 0.5, markersize = 2, label = false)
    contour!(p1, lon_vec, lat_vec, mask_vis, levels=[0.5], color=:black, linewidth=2)
    Plots.savefig(p1, outfile1)

    # V-component anomaly plot
    p2 = heatmap(lon_vec, lat_vec, diff_v_2d,
        title = "V-component Anomaly (Copernicus GT - DIVAnd)", 
        titlefontsize = 8,
        xlabel = "Longitude (°)", 
        ylabel = "Latitude (°)",
        color = :seismic,
        clims = (-1., 1.),
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
        fontsize = 5, size = (600, 600))
    
    scatter!(p2, obs_lon[idxs], obs_lat[idxs], color = :black, alpha = 0.5, markersize = 2, label = false)
    contour!(p2, lon_vec, lat_vec, mask_vis, levels=[0.5], color=:black, linewidth=2)
    Plots.savefig(p2, outfile2)

    # Speed magnitude anomaly plot
    p3 = heatmap(lon_vec, lat_vec, diff_magnitude_2d,
        title = "Speed Anomaly (Copernicus GT - DIVAnd)", 
        titlefontsize = 8,
        xlabel = "Longitude (°)", 
        ylabel = "Latitude (°)",
        color = :seismic,
        clims = (-1., 1.),
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
        fontsize = 5, size = (600, 600))

    scatter!(p3, obs_lon[idxs], obs_lat[idxs], color = :black, alpha = 0.3, markersize = 2, label = false)
    contour!(p3, lon_vec, lat_vec, mask_vis, levels=[0.5], color=:black, linewidth=2)
    Plots.savefig(p3, outfile3)

    # Histogram of differences
    p4 = Plots.histogram([diff_u[:], diff_v[:], diff_magnitude[:]], bins=30,
               labels=["U-anomaly", "V-anomaly", "Speed-anomaly"],
               title="Anomaly distributions",
               xlabel="Velocity anomaly (m/s)",
               ylabel="Frequency",
               alpha=0.7)
    Plots.savefig(p4, outfile4)
end


# Function to calculate correlation between fields
function calculate_correlation(u_interp, v_interp, u_model, v_model)
    """
    Calculate correlation coefficients between interpolated and model velocities.
    """
    # Correlation analysis
    valid_u_mask = .!isnan.(u_interp) .& .!isnan.(u_model)
    valid_v_mask = .!isnan.(v_interp) .& .!isnan.(v_model)
    valid_speed_mask = .!isnan.(u_interp) .& .!isnan.(v_interp) .& .!isnan.(u_model) .& .!isnan.(v_model)
    
    u_corr = cor(u_interp[valid_u_mask], u_model[valid_u_mask])
    v_corr = cor(v_interp[valid_v_mask], v_model[valid_v_mask])
    speed_interp = sqrt.(u_interp.^2 .+ v_interp.^2)
    speed_model = sqrt.(u_model.^2 .+ v_model.^2)
    
    speed_corr = cor(speed_interp[valid_speed_mask], speed_model[valid_speed_mask])
    
    println("U correlation: ", round(u_corr, digits=3))
    println("V correlation: ", round(v_corr, digits=3))
    println("Speed correlation: ", round(speed_corr, digits=3))
end


# Function to calculate vorticity using finite differences
function calculate_vorticity(u, v, lon, lat, nx=nothing, ny=nothing)
    """
    Calculate vorticity (∂v/∂x - ∂u/∂y) using finite differences.
    Works with both 2D arrays and 1D vectors.
    
    For 2D arrays: calculate_vorticity(u_2d, v_2d, lon_2d, lat_2d)
    For 1D vectors: calculate_vorticity(u_vec, v_vec, lon_vec, lat_vec, nx, ny)
    """
    
    # Check if inputs are 1D vectors
    if ndims(u) == 1
        if isnothing(nx) || isnothing(ny)
            error("For 1D vectors, please provide grid dimensions nx and ny")
        end
        
        # Reshape to 2D
        u = reshape(u, ny, nx)
        v = reshape(v, ny, nx)
        lon = reshape(lon, ny, nx)
        lat = reshape(lat, ny, nx)
        return_1d = true
    else
        return_1d = false
    end
    
    ny, nx = size(u)
    vorticity = fill(NaN, size(u))
    
    # Earth radius in meters
    R = 6.371e6
    
    for i in 2:ny-1
        for j in 2:nx-1
            # Skip if any value is NaN
            if isnan(u[i,j]) || isnan(v[i,j]) || 
               isnan(u[i+1,j]) || isnan(u[i-1,j]) ||
               isnan(v[i,j+1]) || isnan(v[i,j-1])
                continue
            end
            
            # Calculate grid spacing in meters
            dx = R * cos(lat[i,j] * π/180) * (lat[i,j+1] - lat[i,j-1]) * π/180
            dy = R * (lon[i+1,j] - lon[i-1,j]) * π/180
            
            # Calculate derivatives
            dvdx = (v[i,j+1] - v[i,j-1]) / (2*dx)
            dudy = (u[i+1,j] - u[i-1,j]) / (2*dy)
            
            vorticity[i,j] = dvdx - dudy
        end
    end
    
    return return_1d ? vec(vorticity) : vorticity
end


# Function to calculate divergence using finite differences
function calculate_divergence(u, v, lon, lat, nx=nothing, ny=nothing)
    """
    Calculate divergence (∂u/∂x + ∂v/∂y) using finite differences.
    Works with both 2D arrays and 1D vectors.
    
    For 2D arrays: calculate_divergence(u_2d, v_2d, lon_2d, lat_2d)
    For 1D vectors: calculate_divergence(u_vec, v_vec, lon_vec, lat_vec, nx, ny)
    """
    
    if ndims(u) == 1
        if isnothing(nx) || isnothing(ny)
            error("For 1D vectors, please provide grid dimensions nx and ny")
        end
        
        u = reshape(u, ny, nx)
        v = reshape(v, ny, nx)
        lon = reshape(lon, ny, nx)
        lat = reshape(lat, ny, nx)
        return_1d = true
    else
        return_1d = false
    end
    
    ny, nx = size(u)
    divergence = fill(NaN, size(u))
    
    R = 6.371e6
    
    for i in 2:ny-1
        for j in 2:nx-1
            if isnan(u[i,j]) || isnan(v[i,j]) || 
               isnan(u[i+1,j]) || isnan(u[i-1,j]) ||
               isnan(v[i,j+1]) || isnan(v[i,j-1])
                continue
            end
            
            dx = R * cos(lat[i,j] * π/180) * (lat[i,j+1] - lat[i,j-1]) * π/180
            dy = R * (lon[i+1,j] - lon[i-1,j]) * π/180
            
            dudx = (u[i,j+1] - u[i,j-1]) / (2*dx)
            dvdy = (v[i+1,j] - v[i-1,j]) / (2*dy)
            
            divergence[i,j] = dudx + dvdy
        end
    end
    
    return return_1d ? vec(divergence) : divergence
end


# Function to calculate NVDI (Normalized Vector Difference Index)
function calculate_nvdi(u1, v1, u2, v2)
    """
    Calculate Normalized Vector Difference Index.
    NVDI = |V1 - V2| / (|V1| + |V2|)
    """
    mag1 = sqrt.(u1.^2 .+ v1.^2)
    mag2 = sqrt.(u2.^2 .+ v2.^2)
    diff_mag = sqrt.((u2 .- u1).^2 .+ (v2 .- v1).^2)
    
    nvdi = diff_mag ./ (mag1 .+ mag2)
    return nvdi
end


# Function to calculate kinetic energy
function calculate_kinetic_energy(u, v)
    """
    Calculate kinetic energy: KE = 0.5 * (u² + v²)
    """
    ke = 0.5 .* (u.^2 .+ v.^2)
    return ke
end


# Function to calculate kinetic energy spectrum via FFT
function kinetic_energy_spectrum(u::Array{Float32,2}, v::Array{Float32,2})
    """
    Calculate 2D kinetic energy spectrum using FFT.
    """
    û = fftshift(fft(u))
    v̂ = fftshift(fft(v))
    E_k = abs2.(û) .+ abs2.(v̂)
    return E_k
end


# Function to calculate isotropic spectrum from 2D wavenumber space
function isotropic_spectrum(E_k::Array{Float32,2})
    """
    Compute isotropic (azimuthally-averaged) energy spectrum from 2D FFT.
    """
    nx, ny = size(E_k)
    kx = fftshift(fftfreq(nx))
    ky = fftshift(fftfreq(ny))
    Kx = repeat(kx, 1, ny)
    Ky = repeat(ky', nx, 1)
    K = sqrt.(Kx.^2 .+ Ky.^2)

    # Flatten and group
    K_flat = vec(K)
    E_flat = vec(E_k)

    # Bin in wavenumber space
    bins = collect(0:0.01:maximum(K_flat))
    spectrum = zeros(length(bins)-1)

    for i in 1:length(bins)-1
        mask = (K_flat .>= bins[i]) .& (K_flat .< bins[i+1])
        if any(mask)
            spectrum[i] = mean(E_flat[mask])
        end
    end

    return bins[1:end-1], spectrum
end


function plot_nvdi_distribution(nvdi, output_file)
    """
    Plot NVDI distribution histogram.
    """
    p = Plots.histogram(nvdi, title="NVDI Index Distribution", 
                        xlabel="NVDI", ylabel="Frequency", alpha=0.7, bins=50)
    Plots.savefig(p, output_file)
end


function plot_kinetic_energy_comparison(ke_model, ke_interp, output_file)
    """
    Plot scatter comparison of kinetic energy between model and interpolation.
    """
    p = Plots.scatter(ke_model, ke_interp,
                  title="Kinetic Energy Comparison",
                  xlabel="KE Ground Truth (m²/s²)",
                  ylabel="KE DIVAnd (m²/s²)",
                  alpha=0.6,
                  markersize=2)
    
    # Add 1:1 reference line
    ke_max = max(maximum(ke_model), maximum(ke_interp))
    Plots.plot!(p, [0., ke_max], [0., ke_max], color=:red, linestyle=:dash)
    Plots.savefig(p, output_file)
end


function plot_vorticity_anomaly(lon_vec, lat_vec, vort_diff_2d, x_min, x_max, y_min, y_max, 
                                 obs_lon, obs_lat, idxs, output_file)
    """
    Plot vorticity anomaly field scaled by 10^5.
    """
    vort_diff_scaled = vort_diff_2d * 1e5
    
    p = heatmap(lon_vec, lat_vec, vort_diff_scaled,
        title = "Vorticity Anomaly (Copernicus GT - DIVAnd)",
        titlefontsize = 8,
        xlabel = "Longitude (°)", 
        ylabel = "Latitude (°)",
        color = :seismic,
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
        fontsize = 5, 
        size = (600, 600),
        colorbar_title = "Vorticity diff (×10⁻5 s⁻¹)",
        colorbar_titlefontsize = 8)
    
    scatter!(p, obs_lon[idxs], obs_lat[idxs], color = :black, alpha = 0.3, markersize = 2, label = false)
    Plots.savefig(p, output_file)
end


function plot_divergence_anomaly(lon_vec, lat_vec, div_diff_2d, x_min, x_max, y_min, y_max,
                                  obs_lon, obs_lat, idxs, output_file)
    """
    Plot divergence anomaly field scaled by 10^5.
    """
    div_diff_scaled = div_diff_2d * 1e5
    
    p = heatmap(lon_vec, lat_vec, div_diff_scaled,
        title = "Divergence Anomaly (Copernicus GT - DIVAnd)",
        titlefontsize = 8,
        xlabel = "Longitude (°)", 
        ylabel = "Latitude (°)",
        color = :seismic,
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
        fontsize = 5, 
        size = (600, 600),
        colorbar_title = "Divergence diff (×10⁻5 s⁻¹)",
        colorbar_titlefontsize = 8)
    
    scatter!(p, obs_lon[idxs], obs_lat[idxs], color = :black, alpha = 0.3, markersize = 2, label = false)
    Plots.savefig(p, output_file)
end
