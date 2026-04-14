"""
create_csv_radar.py
-------------------
Reads RUV radial velocity files from HF radars, converts them to CSV format,
and writes one CSV per snapshot to the specified output directory.
"""

import HFradar_data as hf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Build the radar metadata dictionary for the Catalan Sea network
radar_dict = hf.radars_cat()

# Time range to process
year      = 2026
month     = 1
ini_day   = 14
final_day = 24 + 1  # Exclusive upper bound (processes up to day 24)

# Input directory containing the raw RUV radial files
my_path = "../data/january_2026/radials_10_days/"

# Output directory where the generated CSV files will be saved
out_path_csv = "../data/january_2026/radials_10_days/csv_files_real_radar/"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

# Convert all RUV files in the date range to CSV format
hf.create_all_csv_radar(my_path, radar_dict, year, month, ini_day, final_day, out_path_csv)
