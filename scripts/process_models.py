import os
import datetime
import requests
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- CONFIGURATION ---
# Northeast US Sector
EXTENT = [-82, -66, 37, 48] 

# Directory to save images
OUTPUT_DIR = "output"

def get_latest_run_time():
    """Calculates the latest likely available run (00Z or 12Z)."""
    now = datetime.datetime.utcnow()
    # AIFS/AIGFS usually available 6 hours after run. 
    if now.hour >= 18:
        run = "12"
        date = now
    elif now.hour >= 6:
        run = "00"
        date = now
    else:
        run = "12"
        date = now - datetime.timedelta(days=1)
    
    return date.strftime("%Y%m%d"), run

def download_file(url, local_filename):
    """Helper to download a file with status printing."""
    print(f"Attempting download: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Success: {local_filename}")
                return True
            else:
                print(f"Failed (Status {r.status_code})")
                return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_ecmwf_aifs(date, run, step="24"):
    """Downloads ECMWF AIFS GRIB2 with fallback URLs."""
    # Filename format: YYYYMMDDHH0000-step-oper-fc.grib2
    # Note: Step is formatted as '24h' (no leading zero typically required for ECMWF Open Data)
    filename = f"{date}{run}0000-{step}h-oper-fc.grib2"
    local_file = f"ecmwf_aifs_{date}_{run}_{step}.grib2"

    # Try Primary Source (ECMWF Data Portal) - Updated to 'aifs-single'
    url1 = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    
    # Try Secondary Source (Azure Mirror) - Often more reliable for bots
    url2 = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{filename}"

    if download_file(url1, local_file):
        return local_file
    elif download_file(url2, local_file):
        return local_file
    else:
        print("All ECMWF AIFS download attempts failed.")
        return None

def download_noaa_aigfs(date, run, step="24"):
    """Downloads NOAA GraphCast/AIGFS GRIB2 from S3."""
    # Padding step to 3 digits (e.g., 24 -> 024)
    step_padded = str(step).zfill(3)
    
    # Bucket URL for GraphCast GFS
    url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step_padded}"
    
    local_file = f"noaa_aigfs_{date}_{run}_{step}.grib2"
    
    if download_file(url, local_file):
        return local_file
    else:
        return None

def plot_sector(file_path, model_name, date, run, step):
    """Reads GRIB and plots the Northeast sector."""
    if not os.path.exists(file_path):
        return

    try:
        print(f"Plotting {model_name}...")
        
        # Load Dataset
        if model_name == "ECMWF-AIFS":
            # Filter for 2m temperature (shortName '2t')
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            data_var = ds['2t']
            
        elif model_name == "NOAA-AIGFS":
            # GraphCast often uses 't2m' or '2t'
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            data_var = ds['t2m']

        # Convert Kelvin to Fahrenheit
        data_f = ((data_var - 273.15) * 9/5) + 32

        # Setup Plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        # Add Map Features
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        # Plot Data
        levels = np.arange(-20, 100, 2) 
        cmap = 'coolwarm'
        
        plot = ax.contourf(data_f.longitude, data_f.latitude, data_f, levels=levels, 
                           transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        
        cbar = plt.colorbar(plot, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label("Temperature (°F)")

        # Titles
        valid_time = datetime.datetime.strptime(date, "%Y%m%d") + \
                     datetime.timedelta(hours=int(run)) + \
                     datetime.timedelta(hours=int(step))
        
        plt.title(f"{model_name} | 2m Temperature", loc='left', fontweight='bold', fontsize=12)
        plt.title(f"Init: {date} {run}Z | Valid: {valid_time.strftime('%Y-%m-%d %H:%M')} (F{step})", loc='right', fontsize=10)

        # Save Image
        out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace('-', '_')}_temp_f{step}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved plot to {out_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting {model_name}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    date, run = get_latest_run_time()
    step = "24" # forecast hour
    
    print(f"--- Starting Processing for {date} {run}Z (Step F{step}) ---")

    # 1. ECMWF AIFS (Updated with fallback)
    ecmwf_file = download_ecmwf_aifs(date, run, step)
    if ecmwf_file:
        plot_sector(ecmwf_file, "ECMWF-AIFS", date, run, step)
        os.remove(ecmwf_file)

    # 2. NOAA AIGFS
    noaa_file = download_noaa_aigfs(date, run, step)
    if noaa_file:
        plot_sector(noaa_file, "NOAA-AIGFS", date, run, step)
        os.remove(noaa_file)

if __name__ == "__main__":
    main()
