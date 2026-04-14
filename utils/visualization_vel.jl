using PyPlot
@pyimport cartopy.crs as ccrs
@pyimport cartopy.feature as cfeature

# -----------------------------------------------------------------------
# Module: visualization_vel.jl
# Purpose: Plotting functions for HF radar observations and DIVAnd
#          velocity field reconstructions over the Catalan Sea domain.
#          All functions save figures to disk.
# -----------------------------------------------------------------------


# =================================================================================================
#  OBSERVATION PLOTS
# =================================================================================================


"""
    plot_radials(xi, yi, mask, obs_lon, obs_lat, obs_rad, angle, flag_radar,
                 radar_dict, SCALE, WIDTH, title_name, x_min, x_max, y_min, y_max,
                 outfile_name)

Creates a quiver plot of the radial velocity observations from each HF radar
station. Each station is shown in a distinct colour.

# Arguments
- `xi, yi`        : 2D longitude and latitude grids for the mask
- `mask`          : Land/sea mask (values > 0.5 = ocean)
- `obs_lon`       : Longitudes of the radial observations
- `obs_lat`       : Latitudes of the radial observations
- `obs_rad`       : Radial velocity magnitudes (m/s)
- `angle`         : Bearing angles of each radial (degrees)
- `flag_radar`    : Antenna flag for each observation
- `radar_dict`    : Dictionary mapping radar names to flag integers
- `SCALE, WIDTH`  : Quiver scale and arrow width
- `title_name`    : Figure title
- `x_min, x_max`  : Longitude axis limits
- `y_min, y_max`  : Latitude axis limits
- `outfile_name`  : Output filepath (with extension)
"""
function plot_radials(xi, yi, mask, obs_lon, obs_lat, obs_rad, angle, flag_radar, radar_dict,
                      SCALE, WIDTH, title_name, x_min, x_max, y_min, y_max, outfile_name)

    figure(figsize=(9, 9))

    # Convert bearing angles from degrees to radians for decomposition
    α = angle * pi / 180

    # One colour per radar station
    colors = ["red", "green", "blue", "purple", "gray", "gold"]

    ax = gca()
    q  = nothing

    for i = 1:length(radar_dict)

        radar_name = [k for (k, v) in radar_dict if v == i][1]

        if i in flag_radar

            # Decompose radial velocity into east and north components
            q = ax.quiver(
                obs_lon[flag_radar .== i], obs_lat[flag_radar .== i],
                obs_rad[flag_radar .== i] .* sin.(α[flag_radar .== i]),
                obs_rad[flag_radar .== i] .* cos.(α[flag_radar .== i]),
                color=colors[i], alpha=0.65, scale=SCALE, width=WIDTH,
                label=radar_name * " HF Radar"
            )
        else
            continue
        end
    end

    # Render the ocean and land using the mask
    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], colors=["lightblue"], alpha=0.3)
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper",        alpha=1.0)

    # Reference arrow (0.5 m/s) at a fixed map location
    ax.quiverkey(q, X=2., Y=42., U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.title(title_name)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.legend(loc="upper right", fontsize=10)
    PyPlot.savefig(outfile_name)
end


"""
    plot_radials_uv(xi, yi, mask, obs_lon, obs_lat, u_rad, v_rad, flag_radar,
                    radar_dict, SCALE, WIDTH, title_name, x_min, x_max, y_min,
                    y_max, outfile_name)

Creates a quiver plot of the radial observations expressed as (u, v) velocity
components. Each radar station is shown in a distinct colour.

# Arguments
- `u_rad, v_rad`  : East and north components of the radial velocity (m/s)
(remaining arguments are the same as `plot_radials`)
"""
function plot_radials_uv(xi, yi, mask, obs_lon, obs_lat, u_rad, v_rad, flag_radar, radar_dict,
                          SCALE, WIDTH, title_name, x_min, x_max, y_min, y_max, outfile_name)

    figure(figsize=(9, 9))

    colors = ["red", "green", "blue", "purple", "gray", "gold"]

    ax = gca()
    q  = nothing

    for i = 1:length(radar_dict)

        radar_name = [k for (k, v) in radar_dict if v == i][1]

        # Plot (u, v) components directly for each radar station
        q = ax.quiver(
            obs_lon[flag_radar .== i], obs_lat[flag_radar .== i],
            u_rad[flag_radar .== i],   v_rad[flag_radar .== i],
            color=colors[i], alpha=1.0, scale=SCALE, width=WIDTH,
            label=radar_name * " HF Radar"
        )
    end

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], colors=["lightblue"], alpha=0.3)
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper",        alpha=1.0)

    ax.quiverkey(q, X=2., Y=42., U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.title(title_name)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.legend(loc="upper right", fontsize=10)
    PyPlot.savefig(outfile_name)
end


"""
    plot_vel_total(xi, yi, mask, lon, lat, u, v, pdt, SCALE, WIDTH, title_name,
                   x_min, x_max, y_min, y_max, outfile_name)

Creates a quiver plot of a total velocity field (e.g. from the L3 HF radar
product) overlaid on the domain mask.

# Arguments
- `lon, lat`     : 2D longitude and latitude grids of the velocity field
- `u, v`         : East and north total velocity components (m/s)
- `pdt`          : Decimation factor for the quiver arrows
(remaining arguments same as above)
"""
function plot_vel_total(xi, yi, mask, lon, lat, u, v, pdt, SCALE, WIDTH,
                        title_name, x_min, x_max, y_min, y_max, outfile_name)

    figure(figsize=(9, 9))

    ax = gca()

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], colors=["lightblue"], alpha=0.3)
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper",        alpha=1.0)

    # Total velocity quiver (blue, decimated by pdt)
    Q = ax.quiver(
        lon[1:pdt:end, 1:pdt:end], lat[1:pdt:end, 1:pdt:end],
        u[1:pdt:end, 1:pdt:end],   v[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, color="blue"
    )

    ax.quiverkey(Q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    PyPlot.title(title_name)
    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.savefig(outfile_name)
end


# =================================================================================================
#  DIVAND PLOT
# =================================================================================================


"""
    plot_divand(pdt, xi, yi, uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc,
                eps2_divc, title_name, x_min, x_max, y_min, y_max, mask,
                outfile_name)

Creates a quiver plot of the DIVAnd reconstructed velocity field, annotated
with the simulation parameters used.

# Arguments
- `uri, vri`    : DIVAnd reconstructed east and north velocity components
- `len`         : DIVAnd correlation length scale
- `epsilon2`    : Observation noise-to-signal ratio
- `eps2_bc`     : Boundary condition constraint weight
- `eps2_divc`   : Divergence constraint weight
(remaining arguments same as above)
"""
function plot_divand(pdt, xi, yi, uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc,
                     eps2_divc, title_name, x_min, x_max, y_min, y_max, mask, outfile_name)

    figure(figsize=(9, 9))

    ax = gca()

    # DIVAnd velocity quiver (black, fully opaque)
    q = ax.quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, color="black", alpha=1.
    )

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], colors=["lightblue"], alpha=0.3)
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper",        alpha=1.0)

    ax.quiverkey(q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    # Annotation box showing the DIVAnd simulation parameters
    ax.text(1.7, 42.1,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=8, color="black",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.title(title_name)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.savefig(outfile_name)
end


# =================================================================================================
#  DIVAND + OBSERVATIONS PLOTS
# =================================================================================================


"""
    plot_vel_radial_obs(pdt, xi, yi, obs_lon, obs_lat, obs_rad, angle,
                        flag_radar, radar_dict, uri, vri, SCALE, WIDTH, len,
                        epsilon2, eps2_bc, eps2_divc, title_name, x_min, x_max,
                        y_min, y_max, mask, outfile_name)

Creates a combined quiver plot showing both the radial observations (coloured
by radar station, semi-transparent) and the DIVAnd reconstructed velocity
field (black arrows) on the same axes.
"""
function plot_vel_radial_obs(pdt, xi, yi, obs_lon, obs_lat, obs_rad, angle, flag_radar,
                              radar_dict, uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc,
                              eps2_divc, title_name, x_min, x_max, y_min, y_max, mask,
                              outfile_name)

    figure(figsize=(9, 9))

    ax = gca()

    # Convert bearing angles to radians for vector decomposition
    α = angle * pi / 180

    colors = ["red", "green", "blue", "purple", "gray"]

    # Radial observations (semi-transparent, one colour per station, lowest z-order)
    for i = 1:length(radar_dict)

        radar_name = [k for (k, v) in radar_dict if v == i][1]

        if i in flag_radar
            q = ax.quiver(
                obs_lon[flag_radar .== i], obs_lat[flag_radar .== i],
                obs_rad[flag_radar .== i] .* sin.(α[flag_radar .== i]),
                obs_rad[flag_radar .== i] .* cos.(α[flag_radar .== i]),
                color=colors[i], alpha=0.4, scale=SCALE1, width=WIDTH,
                label=radar_name * " HF Radar", zorder=0
            )
        else
            continue
        end
    end

    # DIVAnd velocity field (on top of observations, higher z-order)
    q = ax.quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, zorder=1
    )

    water_mask = mask .> 0.5
    land_mask  = mask .<= 0.5

    ax.contourf(xi, yi, water_mask, levels=[0.5, 1.0], colors=["lightblue"], alpha=0.3)
    ax.contourf(xi, yi, land_mask,  levels=[0.5, 1.0], cmap="copper",        alpha=1.0)

    ax.quiverkey(q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    # Parameter annotation box
    ax.text(1.2, 43.,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=8, color="black",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.title(title_name)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.savefig(outfile_name)
end


"""
    plot_vel_total_obs(pdt, pdt1, xi, yi, obs_lon, obs_lat, obs_u, obs_v,
                       uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                       title_name, x_min, x_max, y_min, y_max, outfile_name)

Plots total velocity observations (blue, semi-transparent) together with
the DIVAnd reconstructed field (black) on a single axes panel.

# Arguments
- `obs_lon, obs_lat` : 2D grids of observed total velocity positions
- `obs_u, obs_v`     : Observed east and north velocity components
- `pdt`              : Decimation factor for the DIVAnd field
- `pdt1`             : Decimation factor for the observations
"""
function plot_vel_total_obs(pdt, pdt1, xi, yi, obs_lon, obs_lat, obs_u, obs_v,
                             uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                             title_name, x_min, x_max, y_min, y_max, outfile_name)

    ax = gca()

    # Total velocity observations (blue, semi-transparent)
    Q = ax.quiver(
        obs_lon[1:pdt1:end, 1:pdt1:end], obs_lat[1:pdt1:end, 1:pdt1:end],
        obs_u[1:pdt1:end, 1:pdt1:end],   obs_v[1:pdt1:end, 1:pdt1:end],
        scale=SCALE, width=WIDTH, color="blue", alpha=0.7
    )

    # DIVAnd reconstructed field (black, higher z-order)
    q1 = ax.quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, zorder=1
    )

    # Land mask contour fill
    ax.contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.)

    ax.quiverkey(Q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                 color="black", coordinates="data")

    # Parameter annotation box
    ax.text(1.2, 43.,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=8, color="black",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    PyPlot.xlim(x_min, x_max)
    PyPlot.ylim(y_min, y_max)
    PyPlot.title(title_name)
    PyPlot.ylabel("Latitude (°)", fontsize=10)
    PyPlot.xlabel("Longitude (°)", fontsize=10)
    PyPlot.legend(loc="upper right", fontsize=10)
    PyPlot.savefig(outfile_name)
end


"""
    two_plot_total_obs(pdt, pdt1, xi, yi, obs_lon, obs_lat, obs_u, obs_v,
                       uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                       title_name1, title_name2, x_min, x_max, y_min, y_max,
                       mask, outfile_name; obs_point_lon=nothing,
                       obs_point_lat=nothing)

Creates a side-by-side two-panel figure:
  - Left panel  : Total velocity observations (blue arrows)
  - Right panel : DIVAnd reconstructed velocity field (black arrows)

An optional observation point marker can be overlaid on both panels.
"""
function two_plot_total_obs(pdt, pdt1, xi, yi, obs_lon, obs_lat, obs_u, obs_v,
                             uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                             title_name1, title_name2, x_min, x_max, y_min, y_max,
                             mask, outfile_name;
                             obs_point_lon=nothing, obs_point_lat=nothing)

    fig, ax = subplots(1, 2, figsize=(12, 6))

    # ── Left panel: observations ──────────────────────────────────────
    Q = ax[1].quiver(
        obs_lon[1:pdt1:end, 1:pdt1:end], obs_lat[1:pdt1:end, 1:pdt1:end],
        obs_u[1:pdt1:end, 1:pdt1:end],   obs_v[1:pdt1:end, 1:pdt1:end],
        scale=SCALE, width=WIDTH, color="blue", alpha=0.7
    )

    ax[1].quiverkey(Q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                    color="black", coordinates="data")
    ax[1].contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.)

    # Optional observation point marker
    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax[1].plot(obs_point_lon, obs_point_lat, marker="o", markersize=8,
                   color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                   label="Observation point", zorder=10)
    end

    ax[1].set_title(title_name1)
    ax[1].set_xlabel("Longitude (°)", fontsize=10)
    ax[1].set_ylabel("Latitude (°)", fontsize=10)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    # ── Right panel: DIVAnd reconstruction ───────────────────────────
    q1 = ax[2].quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, zorder=1
    )
    ax[2].contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.)

    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax[2].plot(obs_point_lon, obs_point_lat, marker="o", markersize=8,
                   color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                   label="Observation point", zorder=10)
    end

    ax[2].quiverkey(q1, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                    color="black", coordinates="data")
    ax[2].set_title(title_name2)
    ax[2].set_xlabel("Longitude (°)", fontsize=10)
    ax[2].set_ylabel("Latitude (°)", fontsize=10)
    ax[2].set_xlim(x_min, x_max)
    ax[2].set_ylim(y_min, y_max)

    # Parameter annotation box on the right panel
    ax[2].text(1.7, 42.1,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=7, color="black",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    tight_layout()
    savefig(outfile_name)
end


"""
    two_plot_total_obs_totals(pdt, pdt1, xi, yi, obs_lon, obs_lat,
                              lon_totals, lat_totals, obs_u, obs_v, uri, vri,
                              SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                              title_name1, title_name2, x_min, x_max, y_min, y_max,
                              mask, outfile_name; obs_point_lon=nothing,
                              obs_point_lat=nothing)

Creates a side-by-side two-panel figure:
  - Left panel  : Total velocity observations (blue arrows)
  - Right panel : DIVAnd reconstruction (black arrows) with LS total velocity
                  point locations overlaid as red scatter markers

# Additional arguments (vs. two_plot_total_obs)
- `lon_totals, lat_totals` : Locations of the Least Squares total velocity points
"""
function two_plot_total_obs_totals(pdt, pdt1, xi, yi, obs_lon, obs_lat,
                                    lon_totals, lat_totals, obs_u, obs_v,
                                    uri, vri, SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                                    title_name1, title_name2, x_min, x_max, y_min, y_max,
                                    mask, outfile_name;
                                    obs_point_lon=nothing, obs_point_lat=nothing)

    fig, ax = subplots(1, 2, figsize=(12, 6))

    # ── Left panel: observations ──────────────────────────────────────
    Q = ax[1].quiver(
        obs_lon[1:pdt1:end, 1:pdt1:end], obs_lat[1:pdt1:end, 1:pdt1:end],
        obs_u[1:pdt1:end, 1:pdt1:end],   obs_v[1:pdt1:end, 1:pdt1:end],
        scale=SCALE, width=WIDTH, color="blue", alpha=0.7
    )

    ax[1].quiverkey(Q, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                    color="black", coordinates="data")
    ax[1].contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.)

    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax[1].plot(obs_point_lon, obs_point_lat, marker="o", markersize=8,
                   color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                   label="Observation point", zorder=10)
    end

    ax[1].set_title(title_name1)
    ax[1].set_xlabel("Longitude (°)", fontsize=10)
    ax[1].set_ylabel("Latitude (°)", fontsize=10)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    # ── Right panel: DIVAnd reconstruction + LS point locations ──────
    q1 = ax[2].quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, zorder=1
    )
    ax[2].contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.)

    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax[2].plot(obs_point_lon, obs_point_lat, marker="o", markersize=8,
                   color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                   label="Total velocity points", zorder=10)
    end

    # Scatter plot of Least Squares total velocity locations (red, semi-transparent)
    q2 = ax[2].scatter(lon_totals[1:3:end], lat_totals[1:3:end], s=7, c="red",
                       alpha=0.4, edgecolors="black", linewidth=0.5, label="LS", zorder=2)

    ax[2].quiverkey(q1, X=2.5, Y=42, U=0.5, label="0.5 m/s", labelpos="W",
                    color="black", coordinates="data")
    ax[2].set_title(title_name2)
    ax[2].set_xlabel("Longitude (°)", fontsize=10)
    ax[2].set_ylabel("Latitude (°)", fontsize=10)
    ax[2].set_xlim(x_min, x_max)
    ax[2].set_ylim(y_min, y_max)
    PyPlot.legend(loc="lower right", fontsize=7)

    ax[2].text(1.7, 42.1,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=7, color="black",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    tight_layout()
    savefig(outfile_name)
end


"""
    two_plot_total_obs_radial(pdt, pdt1, xi, yi, obs_lon_model, obs_lat_model,
                              obs_u_model, obs_v_model, obs_lon, obs_lat, obs_rad,
                              angle, flag_radar, uri, vri, SCALE, WIDTH, len,
                              epsilon2, eps2_bc, eps2_divc, title_name1, title_name2,
                              x_min, x_max, y_min, y_max, mask, outfile_name;
                              obs_point_lon=nothing, obs_point_lat=nothing)

Creates a side-by-side two-panel figure using an Orthographic map projection:
  - Left panel  : Modelled total velocity field (blue arrows)
  - Right panel : DIVAnd reconstruction (black arrows) + radial observation
                  scatter points (coloured by radar station)

# Additional arguments (vs. two_plot_total_obs)
- `obs_lon_model, obs_lat_model` : 2D grids of the modelled velocity field positions
- `obs_u_model, obs_v_model`     : Modelled velocity components
- `obs_lon, obs_lat`             : Radial observation positions
- `obs_rad`                      : Radial velocity magnitudes (unused for scatter)
- `angle`                        : Bearing angles (unused for scatter)
- `flag_radar`                   : Antenna flag for each radial observation
- `radar_dict`                   : Dictionary mapping radar names to flag integers
"""
function two_plot_total_obs_radial(pdt, pdt1, xi, yi, obs_lon_model, obs_lat_model,
                                    obs_u_model, obs_v_model, obs_lon, obs_lat, obs_rad,
                                    angle, flag_radar, uri, vri,
                                    SCALE, WIDTH, len, epsilon2, eps2_bc, eps2_divc,
                                    title_name1, title_name2,
                                    x_min, x_max, y_min, y_max, mask, outfile_name;
                                    obs_point_lon=nothing, obs_point_lat=nothing)

    # Use an Orthographic projection centred on the Western Mediterranean
    projection = ccrs.Orthographic(central_longitude=0, central_latitude=30)

    fig = figure(figsize=(12, 6))

    # Create both subplots with the Cartopy projection
    ax1 = fig.add_subplot(1, 2, 1, projection=projection)
    ax2 = fig.add_subplot(1, 2, 2, projection=projection)

    # ── Left panel: modelled total velocity ──────────────────────────
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.LAND,  facecolor="#D2B48C", alpha=0.3)
    ax1.add_feature(cfeature.OCEAN, facecolor="#ADD8E6", alpha=0.3)

    ax1.set_extent([x_min, x_max, y_min, y_max])

    # Gridlines (no top or right labels)
    gl1 = ax1.gridlines(draw_labels=true, linewidth=0.5, alpha=0.5, linestyle="--")
    gl1.top_labels   = false
    gl1.right_labels = false

    Q = ax1.quiver(
        obs_lon_model[1:pdt1:end, 1:pdt1:end],
        obs_lat_model[1:pdt1:end, 1:pdt1:end],
        obs_u_model[1:pdt1:end, 1:pdt1:end],
        obs_v_model[1:pdt1:end, 1:pdt1:end],
        scale=SCALE, width=WIDTH, color="blue", alpha=0.7, transform=projection
    )

    # Land mask contour
    ax1.contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.0,
                 transform=projection)

    # Optional observation point marker
    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax1.plot([obs_point_lon], [obs_point_lat], marker="o", markersize=8,
                 color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                 label="Observation point", zorder=10, transform=projection)
    end

    ax1.set_title(title_name1, fontsize=11)

    # ── Right panel: DIVAnd + radial observation scatter ─────────────
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.LAND,  facecolor="#D2B48C", alpha=0.3)
    ax2.add_feature(cfeature.OCEAN, facecolor="#ADD8E6", alpha=0.3)

    ax2.set_extent([x_min, x_max, y_min, y_max])

    gl2 = ax2.gridlines(draw_labels=true, linewidth=0.5, alpha=0.5, linestyle="--")
    gl2.top_labels  = false
    gl2.left_labels = false

    # Scatter the radial observation locations (coloured by radar station)
    colors = ["#E91E63", "#00BCD4", "#FF5722", "#8BC34A", "#9C27B0", "#FFC107"]

    for i = 1:length(radar_dict)
        radar_name = [k for (k, v) in radar_dict if v == i][1]

        if i in flag_radar
            radar_indices    = findall(flag_radar .== i)
            selected_indices = radar_indices[1:8:end]  # Decimate to avoid overplotting

            ax2.scatter(obs_lon[selected_indices], obs_lat[selected_indices],
                        c=colors[i], s=7, alpha=0.4, edgecolors="black",
                        linewidth=0.5, label=radar_name, zorder=2,
                        transform=projection)
        end
    end

    # DIVAnd velocity field
    q1 = ax2.quiver(
        xi[1:pdt:end, 1:pdt:end], yi[1:pdt:end, 1:pdt:end],
        uri[1:pdt:end, 1:pdt:end], vri[1:pdt:end, 1:pdt:end],
        scale=SCALE, width=WIDTH, zorder=1, transform=projection
    )

    ax2.contourf(xi, yi, mask, levels=[0, 0.5], cmap="copper", alpha=1.0,
                 transform=projection)

    if obs_point_lon !== nothing && obs_point_lat !== nothing
        ax2.plot([obs_point_lon], [obs_point_lat], marker="o", markersize=8,
                 color="#00CED1", markeredgecolor="black", markeredgewidth=1.5,
                 label="Observation point", zorder=10, transform=projection)
    end

    ax2.set_title(title_name2, fontsize=11)
    ax2.legend(loc="lower right", fontsize=7)

    # Parameter annotation box (using axes-relative coordinates)
    ax2.text(0.02, 0.98,
        "SIMULATION PARAMETERS:\n" *
        "L = " * string(len) * "\n" *
        "Noise = " * string(epsilon2) * "\n" *
        "Boundary constraint = " * string(eps2_bc) * "\n" *
        "Divergence constraint = " * string(eps2_divc),
        fontsize=7, color="black",
        transform=ax2.transAxes, verticalalignment="top",
        bbox=Dict("facecolor" => "white", "edgecolor" => "black", "boxstyle" => "round,pad=0.5")
    )

    tight_layout()
    savefig(outfile_name, dpi=300, bbox_inches="tight")
end

#=
# NOTE: The alternative PlateCarree-projection version of two_plot_total_obs_radial
# is commented out in the original code and retained here for reference only.
# The active version above uses the Orthographic projection.
=#
