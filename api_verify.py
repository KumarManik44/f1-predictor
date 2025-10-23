import fastf1
import pandas as pd
import os

# Configure FastF1 to use Jolpica API (Ergast replacement)
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}\n")

# Enable caching to speed up data loading
fastf1.Cache.enable_cache(cache_dir)

print("FastF1 version:", fastf1.__version__)
print("API configured successfully!\n")

# Test 1: Load a recent race session (2024 Monaco GP Race)
print("Loading 2024 Monaco GP Race data...")
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

# Display basic race results
results = session.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position', 'Points']]
print("\n2024 Monaco GP Results (Top 10):")
print(results.head(10))

print("\n✅ API connection successful! Ready to proceed.")
