import os
import datetime
import requests
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- CONFIGURATION ---
EXTENT = [-82, -66, 37, 48]  # Northeast US
OUTPUT_DIR = "output"
FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]

def get_latest_complete_run():
    """
    Finds the latest run (00Z or 12Z) that is likely to have full data.
    AI models often lag 5-7 hours behind real-time.
    """
    now = datetime.datetime.utcnow()
    
    # Try to determine the latest likely completed run
    if now.hour >= 18:
        # 12Z should be done by 18Z
        return now.strftime("%Y%m%d"), "12"
    elif now.hour >= 6:
        # 00Z should be done by 06Z
        return now.strftime("%Y%m%d"), "00"
    else:
        # Late night/early morning, use yesterday's 12Z
        prev = now - datetime.timedelta(days=1)
        return prev.strftime("%Y%m%d"), "12"

def download_file(url, local_filename):
    """Downloads a file if it doesn't exist."""
    if os.path.exists(local_filename):
        return True
        
    print(f"  Downloading: {url} ...", end=" ")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Success")
                return True
            else:
                print(f"Failed (Status {r.status_code})")
                return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# --- MODEL DOWNLOADERS ---

def get_ecmwf_aifs(date, run, step):
    """
    ECMWF AIFS (Deterministic)
    Path: /forecasts/{date}/{run}z/aifs-single/0p25/oper/...
    """
    step_str = str(step) # ECMWF uses "24h", "6h", etc.
    filename = f"{date}{run}0000-{step_str}h-oper-fc.grib2"
    local = f"temp_ecmwf_{step:03d}.grib2"
    
    # 1. Try Azure Mirror (often faster/better availability)
    url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    if download_file(url_azure, local): return local

    # 2. Try Main Data Portal
    url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    if download_file(url_main, local): return local

    return None

def get_noaa_aigfs(date, run, step):
    """
    NOAA GraphCast (AIGFS)
    Bucket: noaa-nws-graphcastgfs-pds
    Path: graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step}
    """
    step_padded = f"{step:03d}"
    filename = f"graphcastgfs.t{run}z.pgrb2.0p25.f{step_padded}"
    url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/{filename}"
    local = f"temp_aigfs_{step:03d}.grib2"
    
    if download_file(url, local): return local
    return None

# --- PLOTTING ---

def plot_frame(file_path, model_name, date, run, step):
    """Generates the plot for a single frame."""
    if not file_path: return

    try:
        # Smart variable loader
        # GraphCast uses 't2m', ECMWF often '2t'
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': 't2m'}})
            data = ds['t2m']
        except:
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            data = ds['2t']

        # Convert K to F
        data_f = ((data - 273.15) * 9/5) + 32

        # Create Plot
        fig = plt.figure(figsize=(9, 6))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)

        # Plot Contours
        levels = np.arange(-10, 100, 2)
        cmap = 'coolwarm'
        cf = ax.contourf(data.longitude, data.latitude, data_f, levels=levels, 
                         transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        
        # Labels
        valid_time = datetime.datetime.strptime(date, "%Y%m%d") + \
                     datetime.timedelta(hours=int(run) + step)
        
        plt.title(f"{model_name}", loc='left', fontweight='bold')
        plt.title(f"Init: {date} {run}Z | Valid: {valid_time.strftime('%a %H:%M')} (F{step:03d})", loc='right', fontsize=9)
        
        # Save
        filename = f"{model_name.lower().replace(' ', '_')}_f{step:03d}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Cleanup
        ds.close()

    except Exception as e:
        print(f"  Error plotting {model_name} F{step}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    date, run = get_latest_complete_run()
    print(f"--- Processing Run: {date} {run}Z ---")
    
    # Create metadata file for JS to read
    with open(os.path.join(OUTPUT_DIR, "run_info.json"), "w") as f:
        f.write(f'{{"date": "{date}", "run": "{run}"}}')

    for step in FORECAST_HOURS:
        print(f"\n[Forecast Hour {step:03d}]")
        
        # 1. ECMWF AIFS
        f_ecmwf = get_ecmwf_aifs(date, run, step)
        if f_ecmwf:
            plot_frame(f_ecmwf, "ECMWF AIFS", date, run, step)
            os.remove(f_ecmwf)
        else:
            print("  Skipping ECMWF (Download failed)")

        # 2. NOAA AIGFS (GraphCast)
        f_noaa = get_noaa_aigfs(date, run, step)
        if f_noaa:
            plot_frame(f_noaa, "NOAA AIGFS", date, run, step)
            os.remove(f_noaa)
        else:
            print("  Skipping NOAA (Download failed)")

if __name__ == "__main__":
    main()
