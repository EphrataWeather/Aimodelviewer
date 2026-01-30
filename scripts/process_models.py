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

CONFIG = {
    "ECMWF-AIFS": {
        "base_url": "https://data.ecmwf.int/forecasts",
        "type": "ecmwf_open_data",
        "params": {
            "temperature": {"shortName": "2t", "level": "surface", "label": "2m Temperature (°F)"}
        }
    },
    "NOAA-AIGFS": {
        "bucket": "noaa-nws-graphcastgfs-pds", # Using GraphCast as the AIGFS proxy
        "type": "s3_noaa",
        "params": {
            "temperature": {"shortName": "t2m", "level": "2m", "label": "2m Temperature (°F)"}
        }
    }
}

def get_latest_run_time():
    """Calculates the latest likely available run (00Z or 12Z)."""
    now = datetime.datetime.utcnow()
    # AIFS/AIGFS usually available 4-6 hours after run. 
    # If now is 18:00, we want 12Z. If 08:00, we want 00Z.
    if now.hour >= 16:
        run = "12"
        date = now
    elif now.hour >= 4:
        run = "00"
        date = now
    else:
        run = "12"
        date = now - datetime.timedelta(days=1)
    
    return date.strftime("%Y%m%d"), run

def download_ecmwf_aifs(date, run, step="24"):
    """Downloads ECMWF AIFS GRIB2."""
    # URL Structure: https://data.ecmwf.int/forecasts/20240129/12z/aifs/0p25/oper/20240129120000-24h-oper-fc.grib2
    # Note: Filename format is strict.
    
    base = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs/0p25/oper"
    filename = f"{date}{run}0000-{step}h-oper-fc.grib2"
    url = f"{base}/{filename}"
    
    print(f"Downloading ECMWF AIFS: {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        local_file = f"ecmwf_aifs_{date}_{run}_{step}.grib2"
        with open(local_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_file
    else:
        print(f"Failed to download ECMWF AIFS (Status {r.status_code})")
        return None

def download_noaa_aigfs(date, run, step="24"):
    """Downloads NOAA GraphCast/AIGFS GRIB2 from S3."""
    # URL Structure: https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.20240129/12/forecasts_13_levels/graphcastgfs.t12z.pgrb2.0p25.f024
    
    # Padding step to 3 digits (e.g., 24 -> 024)
    step_padded = str(step).zfill(3)
    
    url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step_padded}"
    
    print(f"Downloading NOAA AIGFS: {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        local_file = f"noaa_aigfs_{date}_{run}_{step}.grib2"
        with open(local_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_file
    else:
        print(f"Failed to download NOAA AIGFS (Status {r.status_code})")
        return None

def plot_sector(file_path, model_name, date, run, step, parameter="temperature"):
    """Reads GRIB and plots the Northeast sector."""
    
    try:
        # Load Dataset. filtering by parameter can speed it up
        if model_name == "ECMWF-AIFS":
            # ECMWF uses '2t' for 2m temp
            ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            data_var = ds['2t']
            data_k = data_var - 273.15 # Kelvin to Celsius
            data_f = (data_k * 9/5) + 32 # Celsius to F
            
        elif model_name == "NOAA-AIGFS":
            # GFS uses 't2m' or '2t' depending on dictionary, usually t2m in cfgrib
            ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            data_var = ds['t2m']
            data_k = data_var - 273.15
            data_f = (data_k * 9/5) + 32

        # Plotting
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        # Features
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        # Contours
        cmap = 'coolwarm'
        # Adjust levels for F (e.g., -20 to 100)
        levels = np.arange(-20, 100, 2) 
        
        plot = ax.contourf(data_f.longitude, data_f.latitude, data_f, levels=levels, 
                           transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        
        cbar = plt.colorbar(plot, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label("Temperature (°F)")

        # Metadata Titles
        valid_time = datetime.datetime.strptime(date, "%Y%m%d") + \
                     datetime.timedelta(hours=int(run)) + \
                     datetime.timedelta(hours=int(step))
        
        plt.title(f"{model_name} | 2m Temperature", loc='left', fontweight='bold', fontsize=12)
        plt.title(f"Init: {date} {run}Z | Valid: {valid_time.strftime('%Y-%m-%d %H:%M')} (F{step})", loc='right', fontsize=10)

        # Save
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

    # 1. ECMWF AIFS
    ecmwf_file = download_ecmwf_aifs(date, run, step)
    if ecmwf_file:
        plot_sector(ecmwf_file, "ECMWF-AIFS", date, run, step)
        os.remove(ecmwf_file) # cleanup

    # 2. NOAA AIGFS
    noaa_file = download_noaa_aigfs(date, run, step)
    if noaa_file:
        plot_sector(noaa_file, "NOAA-AIGFS", date, run, step)
        os.remove(noaa_file)

if __name__ == "__main__":
    main()
