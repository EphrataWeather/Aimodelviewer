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
    
    # AIFS/GEFS data is usually ready ~5-6 hours after the run time.
    if now.hour >= 18:
        run = "12"
        date = now
    elif now.hour >= 6:
        run = "00"
        date = now
    else:
        # If it's early morning (e.g. 02:00 UTC), we want yesterday's 12z
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
    filename = f"{date}{run}0000-{step}h-oper-fc.grib2"
    local = f"ecmwf_aifs_det_{date}_{run}.grib2"
    
    # Try Azure Mirror first
    url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    # Fallback to Main
    url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{filename}"
    
    if download_file(url_azure, local): return local
    if download_file(url_main, local): return local
    return None

def download_ecmwf_aifs_ens(date, run, step):
    """ECMWF AIFS Ensemble (Control Member)."""
    filename = f"{date}{run}0000-{step}h-enfo-cf.grib2"
    local = f"ecmwf_aifs_ens_ctrl_{date}_{run}.grib2"
    
    # Try Azure Mirror
    url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-ens/0p25/enfo/{filename}"
    # Fallback to Main (Added this!)
    url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-ens/0p25/enfo/{filename}"
    
    if download_file(url_azure, local): return local
    if download_file(url_main, local): return local
    return None

def download_noaa_aifs(date, run, step):
    """NOAA AIFS."""
    step_padded = str(step).zfill(3)
    # Corrected URL Pattern check
    url = f"https://noaa-nws-aifs-pds.s3.amazonaws.com/aifs.{date}/{run}/aifs.t{run}z.pgrb2.0p25.f{step_padded}"
    local = f"noaa_aifs_{date}_{run}.grib2"
    
    if download_file(url, local): return local
    return None

def download_gefs_hybrid(date, run, step):
    """NOAA GEFS (Physics Control)."""
    step_padded = str(step).zfill(3)
    url = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{date}/{run}/atmos/pgrb2ap5/gec00.t{run}z.pgrb2a.0p50.f{step_padded}"
    local = f"noaa_gefs_ctrl_{date}_{run}.grib2"
    
    if download_file(url, local): return local
    return None

def open_grib_smart(file_path):
    """
    Tries multiple filter combinations to find the temperature variable.
    Returns: (xarray.DataArray, string_label_of_variable)
    """
    # 1. Try 'heightAboveGround' at level 2 (Standard for GFS/GEFS)
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        if 't2m' in ds: return ds['t2m'], 't2m'
        if '2t' in ds: return ds['2t'], '2t'
        if 'tmp' in ds: return ds['tmp'], 'tmp'
    except Exception:
        pass

    # 2. Try 'surface' (Sometimes used by ECMWF AI models)
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        if '2t' in ds: return ds['2t'], '2t'
        if 't2m' in ds: return ds['t2m'], 't2m'
    except Exception:
        pass

    # 3. Last Resort: Open without filters (Can be slow, but works for single-message files)
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib')
        for var in ['t2m', '2t', 'tmp']:
            if var in ds: return ds[var], var
    except Exception:
        pass

    raise ValueError(f"Could not extract temperature from {file_path}")

def plot_model(file_path, model_label, date, run, step):
    if not file_path or not os.path.exists(file_path):
        print(f"Skipping {model_label} (File not found)")
        return

    try:
        print(f"Processing {model_label}...")
        
        # Use the smart opener
        data_var, var_name = open_grib_smart(file_path)
        print(f"Found variable '{var_name}' in {model_label}")

        # Convert to Fahrenheit
        # Check units: Kelvin is usually > 200.
        vals = data_var.values
        if np.nanmax(vals) > 200:
            data_f = ((data_var - 273.15) * 9/5) + 32
        else:
            data_f = (data_var * 9/5) + 32

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=42))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        levels = np.arange(-20, 100, 2) 
        cmap = 'coolwarm'
        
        cf = ax.contourf(data_var.longitude, data_var.latitude, data_f, levels=levels, 
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

    # 3. NOAA AIFS (Native AI)
    f3 = download_noaa_aifs(date, run, step)
    plot_model(f3, "NOAA-AIFS", date, run, step)
    if f3: os.remove(f3)

    # 4. NOAA GEFS (Physics Control)
    f4 = download_gefs_hybrid(date, run, step)
    plot_model(f4, "NOAA-GEFS-Hybrid", date, run, step)
    if f4: os.remove(f4)

if __name__ == "__main__":
    main()
