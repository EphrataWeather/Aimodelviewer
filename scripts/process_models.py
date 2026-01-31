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

def get_latest_run():
    """Finds the most recent 00Z or 12Z run likely to be available."""
    now = datetime.datetime.utcnow()
    # Data lags: AIFS/AIGFS usually available 6-8 hours after init
    if now.hour >= 19:
        return now.strftime("%Y%m%d"), "12"
    elif now.hour >= 7:
        return now.strftime("%Y%m%d"), "00"
    else:
        prev = now - datetime.timedelta(days=1)
        return prev.strftime("%Y%m%d"), "12"

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        return True
    print(f"  Fetching: {url}")
    try:
        with requests.get(url, stream=True, timeout=45) as r:
            if r.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"  Download error: {e}")
    return False

# --- DATA RETRIEVAL ---

def get_ecmwf_aifs(date, run, step):
    """Downloads ECMWF AIFS (0.25 deg)."""
    filename = f"{date}{run}0000-{step}h-oper-fc.grib2"
    local = f"aifs_{step:03d}.grib2"
    # Try Azure Mirror (Fastest for GitHub Actions)
    url = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    if download_file(url, local): return local
    # Fallback to Main
    url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    if download_file(url_main, local): return local
    return None

def get_noaa_aigfs(date, run, step):
    """Downloads NOAA AIGFS (GraphCast GFS)."""
    step_padded = f"{step:03d}"
    url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step_padded}"
    local = f"aigfs_{step:03d}.grib2"
    if download_file(url, local): return local
    return None

# --- PROCESSING & PLOTTING ---

def open_grib_safely(path):
    """Tries multiple filters to catch variable names like '2t' or 't2m'."""
    # Attempt 1: Standard height-based filter
    for short_name in ['t2m', '2t', 'tmp']:
        try:
            ds = xr.open_dataset(path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': short_name}})
            return ds[short_name]
        except:
            continue
    # Attempt 2: Surface-level filter
    try:
        ds = xr.open_dataset(path, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
        for var in ds.data_vars:
            return ds[var]
    except:
        pass
    return None

def process_frame(path, model_name, date, run, step):
    if not path: return
    
    try:
        data = open_grib_safely(path)
        if data is None: 
            print(f"  Could not find temperature in {path}")
            return

        # Unit Conversion: Kelvin to Fahrenheit
        if data.max() > 200:
            data_f = ((data - 273.15) * 1.8) + 32
        else:
            data_f = (data * 1.8) + 32

        # Plot Setup
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        
        # Color scale for Temperature
        levels = np.arange(-10, 105, 2)
        cf = ax.contourf(data.longitude, data.latitude, data_f, 
                         levels=levels, transform=ccrs.PlateCarree(), 
                         cmap='coolwarm', extend='both')
        
        plt.colorbar(cf, orientation='vertical', pad=0.02, aspect=30, label='Temp (°F)')
        
        # Titling
        valid_time = (datetime.datetime.strptime(date + run, "%Y%m%d%H") + 
                      datetime.timedelta(hours=step))
        plt.title(f"{model_name}", loc='left', fontweight='bold')
        plt.title(f"Init: {date} {run}Z | Valid: {valid_time.strftime('%m/%d %H:%M')} (F{step:03d})", 
                  loc='right', fontsize=9)
        
        # Save Output
        model_id = model_name.lower().replace(" ", "_")
        plt.savefig(f"{OUTPUT_DIR}/{model_id}_f{step:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {model_id}_f{step:03d}.png")

    except Exception as e:
        print(f"  Plotting Error: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    date, run = get_latest_run()
    print(f"Processing Latest Run: {date} {run}Z")
    
    # Metadata for JS
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        f.write(f'{{"date": "{date}", "run": "{run}"}}')

    for step in FORECAST_HOURS:
        print(f"\n--- Forecast Hour {step} ---")
        
        # ECMWF AIFS
        f_ec = get_ecmwf_aifs(date, run, step)
        process_frame(f_ec, "ECMWF AIFS", date, run, step)
        if f_ec: os.remove(f_ec)
        
        # NOAA AIGFS
        f_no = get_noaa_aigfs(date, run, step)
        process_frame(f_no, "NOAA AIGFS", date, run, step)
        if f_no: os.remove(f_no)

if __name__ == "__main__":
    main()
