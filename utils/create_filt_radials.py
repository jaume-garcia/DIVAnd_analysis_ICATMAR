"""
create_filt_radials.py
----------------------
Applies a cross-antenna distance filter to every hourly radial snapshot.

Only radial measurements that lie within `distance_max` km of a radial
from a different antenna are retained. The filtered observations are
written to individual text files, one per snapshot.
"""

import HFradar_data as hf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum search radius (km): a radial is kept only if at least one radial
# from a different antenna exists within this distance.
distance_max = [6.]

# Total number of hourly snapshots to process
n_times = 264

# ---------------------------------------------------------------------------
# Main loop — filter each snapshot independently
# ---------------------------------------------------------------------------

for i in range(n_times):

    # Path to the input CSV file for snapshot i
    file_csv = (
        "../data/january_2026/"
        "radials_10_days/csv_files_radials_medsea/"
        "vel_radar_medsea_snapshot_" + str(i).zfill(3) + ".csv"
    )

    print(file_csv)

    # Read raw radial observations from the CSV:
    #   obs_lon, obs_lat     — geographic position of each radial measurement
    #   u_radar, v_radar     — east/north velocity components (m/s)
    #   angle_bearing        — antenna bearing angle (degrees)
    #   r_vel                — radial velocity magnitude (m/s)
    #   angle_direction      — direction of the radial velocity vector (degrees)
    #   flag_radar           — quality-control flag
    obs_lon, obs_lat, u_radar, v_radar, \
        angle_bearing, r_vel, angle_direction, flag_radar = hf.read_obs_csv(file_csv)

    # Destination path for the filtered output file
    output_file = (
        "../data/january_2026/"
        "radials_10_days/filtered_radials/"
        "filtered_radials" + str(round(distance_max[0])) + "_" + str(i).zfill(3) + ".txt"
    )

    print(output_file)

    # Apply the cross-antenna distance filter and write the retained radials to disk.
    # Radials closer than distance_max[0] km to a radial from a different antenna
    # are considered geometrically constrained and are kept for the reconstruction step.
    hf.create_filtered_radials(
        obs_lon, obs_lat, u_radar, v_radar,
        angle_bearing, r_vel, angle_direction,
        flag_radar, distance_max, output_file
    )
