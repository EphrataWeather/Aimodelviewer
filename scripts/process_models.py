import os
import datetime
import requests
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
EXTENT = [-82, -66, 37, 48]  # Northeast US
OUTPUT_DIR = "output"
FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]

def get_latest_run():
    """Finds the most recent 00Z or 12Z run."""
    now = datetime.datetime.utcnow()
    # AIFS/GraphCast lags ~6 hours behind real time
    if now.hour >= 18: return now.strftime("%Y%m%d"), "12"
    elif now.hour >= 6: return now.strftime("%Y%m%d"), "00"
    else: return (now - datetime.timedelta(days=1)).strftime("%Y%m%d"), "12"

def download_file(url, local_filename):
    """Downloads file. Deletes and retries if file is < 1KB (corrupt)."""
    if os.path.exists(local_filename):
        if os.path.getsize(local_filename) > 1024:
            return True
        else:
            print(f"  [!] Found corrupt file {local_filename}, deleting...")
            os.remove(local_filename)
            
    try:
        print(f"  Downloading: {url}...")
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                return True
            else:
                print(f"    Failed (Status {r.status_code})")
    except Exception as e:
        print(f"    Connection error: {e}")
    return False

# --- DATA FETCHING ---
def fetch_data(date, run, step):
    """Downloads both models for a given step. Returns paths or None."""
    # 1. ECMWF AIFS
    ec_file = f"aifs_{step:03d}.grib2"
    ec_url_azure = f"https://ai4edataeuwest.blob.core.windows.net/ecmwf/{date}/{run}z/aifs-single/0p25/oper/{date}{run}0000-{step}h-oper-fc.grib2"
    ec_url_main = f"https://data.ecmwf.int/forecasts/{date}/{run}z/aifs-single/0p25/oper/{date}{run}0000-{step}h-oper-fc.grib2"
    
    if not download_file(ec_url_azure, ec_file):
        download_file(ec_url_main, ec_file)

    # 2. NOAA AIGFS (GraphCast)
    noaa_file = f"aigfs_{step:03d}.grib2"
    # Note: 'forecasts_13_levels' is the correct folder for AWS Open Data
    noaa_url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step:03d}"
    download_file(noaa_url, noaa_file)

    return (ec_file if os.path.exists(ec_file) else None, 
            noaa_file if os.path.exists(noaa_file) else None)

# --- ROBUST DATA READING ---
def get_var_from_ds(ds, options):
    """Helper to find first matching variable from a list of options."""
    for var in options:
        if var in ds: return ds[var]
    return None

def get_variables(path):
    """
    Robustly extracts Temp (2m), U/V Wind (10m), and Precip.
    Opens the file twice (Surface vs Height) to avoid cfgrib errors.
    """
    data = {'t': None, 'u': None, 'v': None, 'p': None}
    
    # PASS 1: Height Above Ground (Temp at 2m, Wind at 10m)
    try:
        ds_h = xr.open_dataset(path, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
        
        # Temp (Try t2m, 2t, tmp, etc)
        data['t'] = get_var_from_ds(ds_h, ['t2m', '2t', 'tmp', 't'])
        
        # Wind (Try 10u, u10, u, etc)
        data['u'] = get_var_from_ds(ds_h, ['10u', 'u10', 'u', 'ugrd'])
        data['v'] = get_var_from_ds(ds_h, ['10v', 'v10', 'v', 'vgrd'])
        ds_h.close()
    except Exception:
        pass

    # PASS 2: Surface (Precipitation, sometimes Temp is here too)
    try:
        ds_s = xr.open_dataset(path, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        
        # Precip (Total Precipitation or Accumulated Precip)
        data['p'] = get_var_from_ds(ds_s, ['tp', 'apcp', 'p3020', 'precip'])
        
        # Fallback: If Temp wasn't found in Pass 1, check Pass 2
        if data['t'] is None:
            data['t'] = get_var_from_ds(ds_s, ['t2m', '2t', 'tmp'])
            
        ds_s.close()
    except Exception:
        pass

    return data

# --- PLOTTING ---
def plot_panel(ax, data, lons, lats, type, title):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    
    if type == 'temp':
        # Convert K to F
        vals = data if data.max() < 150 else (data - 273.15) * 1.8 + 32
        levels = np.arange(-20, 110, 5)
        cf = ax.contourf(lons, lats, vals, levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree(), extend='both')
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, aspect=40, shrink=0.9, label="Temp (°F)")
        
    elif type == 'wind':
        # Speed in Knots
        u, v = data[0], data[1]
        ws = np.sqrt(u**2 + v**2) * 1.94
        
        cf = ax.contourf(lons, lats, ws, levels=np.arange(5, 60, 5), cmap='BuPu', transform=ccrs.PlateCarree(), extend='max')
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, aspect=40, shrink=0.9, label="Wind Speed (kt)")
        
        # Barbs (subsampled)
        skip = 8
        ax.barbs(lons[::skip, ::skip], lats[::skip, ::skip], 
                 u[::skip, ::skip] * 1.94, v[::skip, ::skip] * 1.94, 
                 length=4.5, pivot='middle', linewidth=0.4, transform=ccrs.PlateCarree())

    elif type == 'precip':
        # Rain vs Snow Estimation
        precip_mm = data[0]
        t2m_k = data[1]
        
        # Handle units (some models use mm/s, others mm/6h)
        # Assuming accumulation. If values are tiny (<0.01 max), it might be rate.
        if precip_mm.max() < 0.1: precip_mm = precip_mm * 3600 * 6 # crude conversion attempt if rate
        
        mask = precip_mm > 0.1
        if np.any(mask):
            is_snow = (t2m_k < 273.15) & mask
            is_rain = (t2m_k >= 273.15) & mask
            
            # Plot Snow (Blue)
            if np.any(is_snow):
                ax.contourf(lons, lats, np.where(is_snow, precip_mm, np.nan), 
                           levels=[0.1, 1, 2.5, 5, 10, 25], colors=['#b4d7ff', '#6facff', '#005ce6', '#000099'], 
                           transform=ccrs.PlateCarree())
            # Plot Rain (Green)
            if np.any(is_rain):
                ax.contourf(lons, lats, np.where(is_rain, precip_mm, np.nan), 
                           levels=[0.1, 1, 2.5, 5, 10, 25], colors=['#c2f0c2', '#75e075', '#2db32d', '#006400'], 
                           transform=ccrs.PlateCarree())

    ax.set_title(title, fontsize=9, fontweight='bold')

def generate_plot(args):
    """Worker function."""
    path, model_name, date, run, step = args
    if not path: return

    try:
        vars = get_variables(path)
        
        # Check if we at least have Temperature
        if vars['t'] is None:
            print(f"  [Error] {model_name} F{step}: Could not find Temp variable.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), subplot_kw={'projection': ccrs.LambertConformal()})
        
        # 1. Temperature
        plot_panel(axs[0], vars['t'], vars['t'].longitude, vars['t'].latitude, 'temp', "2m Temperature")
        
        # 2. Wind (if available)
        if vars['u'] is not None and vars['v'] is not None:
            plot_panel(axs[1], (vars['u'], vars['v']), vars['u'].longitude, vars['u'].latitude, 'wind', "10m Wind & Barbs")
        else:
            axs[1].text(0.5, 0.5, "Wind Data Missing", transform=axs[1].transAxes, ha='center')

        # 3. Precip (if available)
        if vars['p'] is not None:
            plot_panel(axs[2], (vars['p'], vars['t']), vars['p'].longitude, vars['p'].latitude, 'precip', "Precip Type (Rain/Snow)")
        else:
            axs[2].text(0.5, 0.5, "Precip Data Missing", transform=axs[2].transAxes, ha='center')

        # Main Title
        valid = (datetime.datetime.strptime(date+run, "%Y%m%d%H") + datetime.timedelta(hours=step))
        plt.suptitle(f"{model_name} | F{step:03d} | Valid: {valid.strftime('%a %m/%d %H:%M Z')}", fontsize=14, y=0.98)
        
        # Save
        outfile = f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_f{step:03d}.png"
        plt.savefig(outfile, bbox_inches='tight', dpi=90)
        plt.close()
        print(f"  [OK] Generated {outfile}")

    except Exception as e:
        print(f"  [Fail] {model_name} F{step}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    date, run = get_latest_run()
    print(f"--- Processing Run: {date} {run}Z ---")
    
    with open(f"{OUTPUT_DIR}/run_info.json", "w") as f:
        f.write(f'{{"date": "{date}", "run": "{run}"}}')

    # Prepare Tasks
    tasks = []
    print("Fetching files...")
    for step in FORECAST_HOURS:
        f_ec, f_no = fetch_data(date, run, step)
        if f_ec: tasks.append((f_ec, "ECMWF AIFS", date, run, step))
        if f_no: tasks.append((f_no, "NOAA AIGFS", date, run, step))

    print(f"Starting generation for {len(tasks)} frames...")
    
    # Run Parallel (Max 4 workers to match standard GitHub Action runners)
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(generate_plot, tasks)

    print("Cleaning up grib files...")
    for f in os.listdir("."):
        if f.endswith(".grib2"): 
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    main()
