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
    if now.hour >= 19: return now.strftime("%Y%m%d"), "12"
    elif now.hour >= 7: return now.strftime("%Y%m%d"), "00"
    else: return (now - datetime.timedelta(days=1)).strftime("%Y%m%d"), "12"

def download_file(url, local_filename):
    if os.path.exists(local_filename): return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                return True
    except: pass
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

    # 2. NOAA AIGFS
    noaa_file = f"aigfs_{step:03d}.grib2"
    noaa_url = f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{date}/{run}/forecasts_13_levels/graphcastgfs.t{run}z.pgrb2.0p25.f{step:03d}"
    download_file(noaa_url, noaa_file)

    return (ec_file if os.path.exists(ec_file) else None, 
            noaa_file if os.path.exists(noaa_file) else None)

# --- DATA EXTRACTION (Optimized) ---
def get_variables(path):
    """
    Extracts Temp (2m), U/V Wind (10m), and Total Precip.
    Optimized to minimize file reads.
    """
    ds_sfc = None
    ds_wind = None
    
    data = {}

    try:
        # 1. Open Surface Level (Temp, Precip)
        # ECMWF uses '2t'/'tp', NOAA uses 't2m'/'apcp' (or 'tp')
        # We try to open strictly to avoid "heterogeneous" errors
        try:
            ds_sfc = xr.open_dataset(path, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        except:
            # Fallback for GFS which puts t2m at 'heightAboveGround' = 2
             ds_sfc = xr.open_dataset(path, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})

        # Extract Temp
        if 't2m' in ds_sfc: data['t'] = ds_sfc['t2m']
        elif '2t' in ds_sfc: data['t'] = ds_sfc['2t']
        elif 'tmp' in ds_sfc: data['t'] = ds_sfc['tmp']
        
        # Extract Precip (Total Precip)
        # Sometimes Precip is in a different message, so we might need a separate open
        try:
            ds_pcp = xr.open_dataset(path, engine='cfgrib', 
                                     backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
            data['p'] = ds_pcp['tp']
        except:
            try:
                ds_pcp = xr.open_dataset(path, engine='cfgrib', 
                                        backend_kwargs={'filter_by_keys': {'shortName': 'apcp'}})
                data['p'] = ds_pcp['apcp']
            except:
                data['p'] = None # No precip found

        # 2. Open Wind Level (10m)
        try:
            ds_wind = xr.open_dataset(path, engine='cfgrib', 
                                      backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            if '10u' in ds_wind:
                data['u'], data['v'] = ds_wind['10u'], ds_wind['10v']
            elif 'u10' in ds_wind:
                data['u'], data['v'] = ds_wind['u10'], ds_wind['v10']
            elif 'ugrd' in ds_wind:
                data['u'], data['v'] = ds_wind['ugrd'], ds_wind['vgrd']
            elif 'u' in ds_wind:
                data['u'], data['v'] = ds_wind['u'], ds_wind['v']
        except:
            data['u'], data['v'] = None, None

    except Exception as e:
        print(f"Read error {path}: {e}")
        
    return data

# --- PLOTTING ---
def plot_panel(ax, data, lons, lats, type, title):
    """Helper to plot a single panel (Temp, Wind, or Precip)."""
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    
    if type == 'temp':
        # Fahrenheit
        vals = data if data.max() < 150 else (data - 273.15) * 1.8 + 32
        levels = np.arange(-20, 110, 5)
        cf = ax.contourf(lons, lats, vals, levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree(), extend='both')
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, aspect=50, shrink=0.8, label="Temp (°F)")
        
    elif type == 'wind':
        # Knots (approx m/s * 1.94)
        u, v = data[0], data[1]
        ws = np.sqrt(u**2 + v**2) * 1.94
        
        # Speed shading
        cf = ax.contourf(lons, lats, ws, levels=np.arange(5, 60, 5), cmap='BuPu', transform=ccrs.PlateCarree())
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, aspect=50, shrink=0.8, label="Wind Speed (kt)")
        
        # Barbs (subsample to avoid clutter)
        skip = 8
        ax.barbs(lons[::skip, ::skip], lats[::skip, ::skip], 
                 u[::skip, ::skip] * 1.94, v[::skip, ::skip] * 1.94, 
                 length=5, pivot='middle', linewidth=0.5, transform=ccrs.PlateCarree())

    elif type == 'precip':
        # Precip Type Estimation
        # data[0] = precip_amount (mm), data[1] = t2m (K)
        precip_mm = data[0]
        t2m_k = data[1]
        
        # Mask out zero precip
        mask = precip_mm > 0.1
        
        if np.any(mask):
            # Define Rain/Snow mask based on Freezing Line (273.15 K)
            is_snow = (t2m_k < 273.15) & mask
            is_rain = (t2m_k >= 273.15) & mask
            
            # Plot Snow (Blues)
            if np.any(is_snow):
                ax.contourf(lons, lats, np.where(is_snow, precip_mm, np.nan), 
                           levels=[0.1, 1, 2.5, 5, 10, 20], colors=['#b4d7ff', '#6facff', '#005ce6'], 
                           transform=ccrs.PlateCarree())
            
            # Plot Rain (Greens)
            if np.any(is_rain):
                ax.contourf(lons, lats, np.where(is_rain, precip_mm, np.nan), 
                           levels=[0.1, 1, 2.5, 5, 10, 20], colors=['#c2f0c2', '#75e075', '#2db32d'], 
                           transform=ccrs.PlateCarree())
        
        # Add legend manually
        import matplotlib.patches as mpatches
        r_patch = mpatches.Patch(color='#75e075', label='Rain')
        s_patch = mpatches.Patch(color='#6facff', label='Snow')
        ax.legend(handles=[r_patch, s_patch], loc='lower right', fontsize=8)

    ax.set_title(title, fontsize=10, fontweight='bold')

def generate_plot(args):
    """Worker function for parallel processing."""
    path, model_name, date, run, step = args
    if not path: return

    try:
        vars = get_variables(path)
        if 't' not in vars: return

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.LambertConformal()})
        
        # Panel 1: Temperature
        plot_panel(axs[0], vars['t'], vars['t'].longitude, vars['t'].latitude, 'temp', "2m Temperature")
        
        # Panel 2: Wind
        if vars['u'] is not None:
            plot_panel(axs[1], (vars['u'], vars['v']), vars['u'].longitude, vars['u'].latitude, 'wind', "10m Wind & Barbs")
        
        # Panel 3: Precip Type
        if vars['p'] is not None:
            plot_panel(axs[2], (vars['p'], vars['t']), vars['p'].longitude, vars['p'].latitude, 'precip', "Precip Type (Simulated)")
        else:
            axs[2].text(0.5, 0.5, "No Precip Data", transform=axs[2].transAxes, ha='center')

        # Main Title
        valid = (datetime.datetime.strptime(date+run, "%Y%m%d%H") + datetime.timedelta(hours=step))
        plt.suptitle(f"{model_name} | F{step:03d} | Valid: {valid.strftime('%a %H:%M Z')}", fontsize=16, y=1.05)
        
        outfile = f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_f{step:03d}.png"
        plt.savefig(outfile, bbox_inches='tight', dpi=90)
        plt.close()
        print(f"Generated {outfile}")

    except Exception as e:
        print(f"Failed {model_name} F{step}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    date, run = get_latest_run()
    print(f"--- Run: {date} {run}Z ---")
    
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        f.write(f'{{"date": "{date}", "run": "{run}"}}')

    # Prepare download list
    tasks = []
    for step in FORECAST_HOURS:
        f_ec, f_no = fetch_data(date, run, step)
        if f_ec: tasks.append((f_ec, "ECMWF AIFS", date, run, step))
        if f_no: tasks.append((f_no, "NOAA AIGFS", date, run, step))

    # Parallel Plotting (Speeds up generation by 2x-4x)
    # GitHub runners usually have 2-4 cores
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(generate_plot, tasks)

    # Cleanup
    print("Cleaning up grib files...")
    for f in os.listdir("."):
        if f.endswith(".grib2"): os.remove(f)

if __name__ == "__main__":
    main()
