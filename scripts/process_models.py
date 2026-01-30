import os
import datetime
import requests
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- CONFIGURATION ---
EXTENT = [-82, -66, 37, 48] # Northeast US
OUTPUT_DIR = "output"

def get_run_times():
    """Determines the target run date/cycle based on current UTC time."""
    now = datetime.datetime.utcnow()
    
    # Logic: Data usually available ~5-6 hours after run
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
    """Robust downloader with timeout."""
    if os.path.exists(local_filename):
        print(f"File already exists: {local_filename}")
        return True
        
    print(f"Attempting: {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
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

# --- DOWNLOADERS ---

def download_ecmwf_aifs(date, run, step):
    """ECMWF AIFS (Deterministic High-Res)."""
    # Note: Using 'aifs-single' for deterministic
    filename = f"{date}{run}0000-{step}h-oper-fc.grib2"
    local = f"ecmwf_aifs_det_{date}_{run}.grib2"
    
    # Try Azure Mirror first (faster/more reliable for bots)
    url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    
    if download_file(url_azure, local): return local
    if download_file(url_main, local): return local
    return None

def download_ecmwf_aifs_ens(date, run, step):
    """ECMWF AIFS Ensemble (Control Member)."""
    # Using 'aifs-ens' stream. We grab the Control (cf) to save bandwidth vs downloading 50 members.
    filename = f"{date}{run}0000-{step}h-enfo-cf.grib2"
    local = f"ecmwf_aifs_ens_ctrl_{date}_{run}.grib2"
    
    url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-ens/0p25/enfo/{filename}"
    
    if download_file(url_azure, local): return local
    return None

def download_noaa_aifs(date, run, step):
    """NOAA AIFS (The NOAA-developed AI model)."""
    # Bucket: noaa-nws-aifs-pds
    step_padded = str(step).zfill(3)
    # File pattern: aifs.t00z.pgrb2.0p25.f024
    url = f"https://noaa-nws-aifs-pds.s3.amazonaws.com/aifs.{date}/{run}/aifs.t{run}z.pgrb2.0p25.f{step_padded}"
    local = f"noaa_aifs_{date}_{run}.grib2"
    
    if download_file(url, local): return local
    return None

def download_gefs_hybrid(date, run, step):
    """NOAA GEFS (Physics Ensemble - Control Member)."""
    # Bucket: noaa-gefs-pds
    # We download the control member (gec00) for a fair comparison with AI Controls
    step_padded = str(step).zfill(3)
    # File: gec00.t00z.pgrb2a.0p50.f024
    url = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{date}/{run}/atmos/pgrb2ap5/gec00.t{run}z.pgrb2a.0p50.f{step_padded}"
    local = f"noaa_gefs_ctrl_{date}_{run}.grib2"
    
    if download_file(url, local): return local
    return None

def get_temp_var(ds):
    """Helper to find temperature variable regardless of naming convention."""
    # Priority: t2m (standard), 2t (ecmwf code), tmp (ncep sometimes)
    if 't2m' in ds:
        return ds['t2m']
    elif '2t' in ds:
        return ds['2t']
    elif 'tmp' in ds: # Sometimes standard GFS uses 'tmp'
        return ds['tmp']
    else:
        raise ValueError(f"Could not find temp variable. Available: {list(ds.keys())}")

def plot_model(file_path, model_label, date, run, step):
    if not file_path or not os.path.exists(file_path):
        print(f"Skipping {model_label} (File not found)")
        return

    try:
        print(f"Processing {model_label}...")
        
        # Open Dataset with filter to speed up reading
        # We try to filter for both common keys to ensure we get the data
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': 't2m'}})
        except:
            # Fallback if t2m filter fails, try 2t
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'shortName': '2t'}})

        data_var = get_temp_var(ds)
        
        # Convert to Fahrenheit
        # Check units: if > 200, assume Kelvin. If < 100, assume Celsius (unlikely in GRIB2)
        if data_var.max() > 200:
            data_f = ((data_var - 273.15) * 9/5) + 32
        else:
            data_f = (data_var * 9/5) + 32

        # Plot
        fig = plt.figure(figsize=(10, 8)) # Slightly smaller to save generation time
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        levels = np.arange(-20, 100, 2) 
        cmap = 'coolwarm'
        
        cf = ax.contourf(data_f.longitude, data_f.latitude, data_f, levels=levels, 
                         transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        
        plt.colorbar(cf, orientation='vertical', pad=0.02, shrink=0.7, label="Temp (°F)")

        # Titles
        valid = datetime.datetime.strptime(date, "%Y%m%d") + datetime.timedelta(hours=int(run)+int(step))
        plt.title(f"{model_label}", loc='left', fontweight='bold', fontsize=12)
        plt.title(f"Init: {date} {run}Z | F{step}", loc='right', fontsize=10)

        # Output
        safe_name = model_label.lower().replace(" ", "_").replace("-", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        print(f"Generated: {out_path}")
        
        ds.close()
        plt.close()

    except Exception as e:
        print(f"Error plotting {model_label}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    date, run = get_run_times()
    step = "24"
    
    print(f"--- RUNNING FOR {date} {run}Z ---")

    # 1. ECMWF AIFS (Deterministic)
    f1 = download_ecmwf_aifs(date, run, step)
    plot_model(f1, "ECMWF-AIFS-Det", date, run, step)
    if f1: os.remove(f1)

    # 2. ECMWF AIFS (Ensemble Control)
    f2 = download_ecmwf_aifs_ens(date, run, step)
    plot_model(f2, "ECMWF-AIFS-Ens-Ctrl", date, run, step)
    if f2: os.remove(f2)

    # 3. NOAA AIFS (NOAA's AI Model)
    f3 = download_noaa_aifs(date, run, step)
    plot_model(f3, "NOAA-AIFS", date, run, step)
    if f3: os.remove(f3)

    # 4. NOAA GEFS (Hybrid/Physics Control)
    f4 = download_gefs_hybrid(date, run, step)
    plot_model(f4, "NOAA-GEFS-Hybrid", date, run, step)
    if f4: os.remove(f4)

if __name__ == "__main__":
    main()
