#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caspar_process_refactored.py

1) Inspect the earliest and latest NetCDF files in the Caspar folder (init = 20210604 to 20210705).
   Print their contents (dimensions, coordinates, sample time values).

2) Extract forecasts with lead times 24h, 72h, 168h, and 240h whose
   valid times fall between 2021-06-05 and 2021-07-05 (inclusive),
   for two locations: Vancouver and Lytton.

3) Load ERA5 reanalysis data for ground truth comparison.

4) Save the cleaned time-series DataFrame to CSV.

5) Aggregate to daily mean & max for each (location, lead_h) and print summary.

6) Plot daily mean and daily max temperature by lead time with ERA5 comparison,
   with formatted date axis, and save plots.

7) Create maps showing the first forecasted timestamp for each lead time.

Usage:
    Modify DATA_DIR and OUTPUT_DIR below, then run:
        python caspar_process_refactored.py
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ───── CONFIGURATION ───────────────────────────────────────────────────
DATA_DIR   = Path("caspar")       # directory with .nc files
ERA5_DIR   = Path("data")         # directory with ERA5 data (same as Aurora)
OUTPUT_DIR = Path("result")       # where CSVs and plots go
PLOT_DIR   = OUTPUT_DIR / "plots"

# Forecast settings
POINTS      = {"Vancouver": {"lat": 49.2827, "lon": -123.1207},
              "Lytton":    {"lat": 50.2333, "lon": -121.7667}}
LEAD_HOURS  = [24, 72, 168, 240]
VALID_START = np.datetime64("2021-06-25")  # Updated to match ERA5 data
VALID_END   = np.datetime64("2021-07-05")  # Updated to match ERA5 data

# prepare output
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ───── LOGGING SETUP ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ───── FUNCTIONS ───────────────────────────────────────────────────────

def inspect_files(nc_files):
    earliest, latest = nc_files[0], nc_files[-1]
    for f in (earliest, latest):
        with xr.open_dataset(f) as ds:
            logging.info("Inspecting %s", f.name)
            logging.info(ds)


def pick_temp_var(ds):
    for v in ds.data_vars:
        lname = v.lower()
        if any(key in lname for key in ("tmp", "tt", "temperature", "t2m")):
            return v
    return list(ds.data_vars)[0]


def inspect_era5_files():
    """Inspect ERA5 data structure to understand variable names and dimensions"""
    era5_files = [
        ERA5_DIR / "ERA5_2021_06" / "data_stream-oper_stepType-instant.nc",
        ERA5_DIR / "ERA5_2021_07" / "data_stream-oper_stepType-instant.nc"
    ]
    
    for era5_file in era5_files:
        if era5_file.exists():
            logging.info("=" * 60)
            logging.info("Inspecting ERA5 file: %s", era5_file.name)
            logging.info("=" * 60)
            
            with xr.open_dataset(era5_file) as ds:
                logging.info("Dataset info:")
                logging.info(ds)
                logging.info("\nDimensions:")
                for dim, size in ds.dims.items():
                    logging.info(f"  {dim}: {size}")
                logging.info("\nCoordinates:")
                for coord in ds.coords:
                    logging.info(f"  {coord}: shape {ds.coords[coord].shape}")
                logging.info("\nData variables:")
                for var in ds.data_vars:
                    logging.info(f"  {var}: shape {ds.data_vars[var].shape}")
                    if hasattr(ds.data_vars[var], 'attrs'):
                        logging.info(f"    attrs: {ds.data_vars[var].attrs}")
                
                # Show sample values for valid_time dimension
                if 'valid_time' in ds.coords:
                    times = ds.valid_time.values
                    logging.info(f"\nSample valid_time values (first 5):")
                    for i, t in enumerate(times[:5]):
                        logging.info(f"  valid_time[{i}] = {pd.to_datetime(t)}")
                    logging.info(f"Sample valid_time values (last 5):")
                    for i, t in enumerate(times[-5:]):
                        logging.info(f"  valid_time[{len(times)-5+i}] = {pd.to_datetime(t)}")
                else:
                    logging.info("\nNo 'valid_time' coordinate found. Available coordinates:")
                    for coord in ds.coords:
                        if ds.coords[coord].size > 0:
                            logging.info(f"  {coord}: {ds.coords[coord].values[:3]}...")
                        else:
                            logging.info(f"  {coord}: scalar value")
        else:
            logging.warning("ERA5 file not found: %s", era5_file)


def extract_era5_data():
    """Extract ERA5 reanalysis data for ground truth comparison using the same approach as Aurora"""
    records = []
    
    # Use the same ERA5 data as Aurora script
    try:
        # Load ERA5 surface data for June and July (same as Aurora)
        surf_ds = xr.open_mfdataset([
            ERA5_DIR/'2021-06-surface-level.nc',
            ERA5_DIR/'2021-07-surface-level.nc'
        ], combine='by_coords', engine='netcdf4')
        
        logging.info("Loaded ERA5 surface data successfully")
        
        # Get temperature data and convert to Celsius (same as Aurora)
        t2m = surf_ds['t2m'].values - 273.15  # Convert from Kelvin to Celsius
        times = surf_ds.valid_time.values.astype('datetime64[h]')
        
        # Process each location (same approach as Aurora)
        for name, loc in POINTS.items():
            logging.info(f"Processing ERA5 data for {name}")
            
            # Find nearest grid point (same as Aurora)
            lons = surf_ds.longitude.values
            lats = surf_ds.latitude.values
            lon_mod = loc['lon'] % 360
            i_lon = np.abs(lons - lon_mod).argmin()
            i_lat = np.abs(lats - loc['lat']).argmin()
            
            # Extract temperature time series for this location (same as Aurora)
            temp_series = t2m[:, i_lat, i_lon]
            
            # Convert to pandas Series for resampling (same as Aurora)
            dt_idx = pd.to_datetime(times.astype('datetime64[s]'))
            ser = pd.Series(temp_series, index=dt_idx)
            
            # Resample to daily max and mean (same as Aurora)
            daily_max = ser.resample('1D').max()
            daily_mean = ser.resample('1D').mean()
            
            # Filter to our date range
            date_range = pd.date_range(start=VALID_START, end=VALID_END, freq='D')
            daily_max_filtered = daily_max.reindex(date_range)
            daily_mean_filtered = daily_mean.reindex(date_range)
            
            # Add to records
            for date, max_temp, mean_temp in zip(date_range, daily_max_filtered.values, daily_mean_filtered.values):
                if not pd.isna(max_temp):
                    records.append({
                        'location': name,
                        'valid_time': date,
                        'date': date,
                        'temp': max_temp,
                        'metric': 'daily_max',
                        'source': 'ERA5'
                    })
                if not pd.isna(mean_temp):
                    records.append({
                        'location': name,
                        'valid_time': date,
                        'date': date,
                        'temp': mean_temp,
                        'metric': 'daily_mean',
                        'source': 'ERA5'
                    })
        
        surf_ds.close()
        
    except FileNotFoundError:
        logging.error("ERA5 data files not found in data directory. Please ensure the following files exist:")
        logging.error("  - data/2021-06-surface-level.nc")
        logging.error("  - data/2021-07-surface-level.nc")
        return pd.DataFrame(records)
    
    df_era5 = pd.DataFrame(records)
    logging.info("Extracted %d ERA5 records", len(df_era5))
    return df_era5


def extract_timeseries(nc_files):
    records = []
    debug_printed = 0
    for f in nc_files:
        with xr.open_dataset(f) as ds:
            var = pick_temp_var(ds)
            da = ds[var]
            if getattr(da, 'units', '').lower().startswith(('k', 'kelvin')):
                da = da - 273.15
            init = pd.to_datetime(f.stem, format="%Y%m%d%H")
            leads = ((ds.time - np.datetime64(init)) / np.timedelta64(1, 'h')).astype(int)
            times = ds.time.values
            mask = np.isin(leads, LEAD_HOURS) & \
                   (times >= VALID_START) & (times <= VALID_END + np.timedelta64(23, 'h'))
            for t, lh in zip(times[mask], leads.values[mask]):
                arr2d = da.sel(time=t).squeeze()
                lat2d, lon2d = ds.lat.values, ds.lon.values % 360
                for name, loc in POINTS.items():
                    dist = np.abs(lat2d - loc['lat']) + np.abs(lon2d - (loc['lon'] % 360))
                    i, j = np.unravel_index(np.argmin(dist), dist.shape)
                    valid_time = pd.to_datetime(t)
                    # Debug print for the first few records
                    if debug_printed < 10:
                        logging.info(f"Forecast: location={name}, valid_time={valid_time}, lead_h={lh}, init_time={init}, file={f.name}")
                        assert init == valid_time - pd.Timedelta(hours=int(lh)), f"Init time mismatch: {init} != {valid_time} - {lh}h"
                        debug_printed += 1
                    records.append({
                        'location':   name,
                        'valid_time': valid_time,
                        'date':       valid_time.floor('D'),
                        'lead_h':     int(lh),
                        'temp':       float(arr2d.isel(rlat=i, rlon=j).item()),
                        'source':     'CaSPAr'
                    })
    df = pd.DataFrame(records)
    logging.info("Extracted %d CaSPAr records", len(df))
    return df


def save_csv(df_caspar, df_era5):
    # Save CaSPAr data
    out_caspar = OUTPUT_DIR / 'caspar_cleaned_timeseries.csv'
    df_caspar.to_csv(out_caspar, index=False)
    logging.info("Saved CaSPAr CSV: %s", out_caspar)
    
    # Save ERA5 data
    out_era5 = OUTPUT_DIR / 'era5_ground_truth.csv'
    df_era5.to_csv(out_era5, index=False)
    logging.info("Saved ERA5 CSV: %s", out_era5)


def aggregate_daily(df_caspar, df_era5):
    # Aggregate CaSPAr data
    df_caspar['date'] = pd.to_datetime(df_caspar['date'])
    daily_caspar = (df_caspar.groupby(['location','lead_h','date'])['temp']
                    .agg(daily_mean='mean', daily_max='max')
                    .reset_index())
    daily_caspar['source'] = 'CaSPAr'
    
    # Handle ERA5 data - check if it has the new structure with 'metric' column
    if 'metric' in df_era5.columns:
        # New structure: separate rows for daily_max and daily_mean
        df_era5['date'] = pd.to_datetime(df_era5['date'])
        daily_era5 = df_era5.pivot_table(
            index=['location', 'date'], 
            columns='metric', 
            values='temp'
        ).reset_index()
        daily_era5['source'] = 'ERA5'
        daily_era5['lead_h'] = 0  # ERA5 has no lead time
    else:
        # Old structure: aggregate from hourly data
        df_era5['date'] = pd.to_datetime(df_era5['date'])
        daily_era5 = (df_era5.groupby(['location','date'])['temp']
                      .agg(daily_mean='mean', daily_max='max')
                      .reset_index())
        daily_era5['source'] = 'ERA5'
        daily_era5['lead_h'] = 0  # ERA5 has no lead time
    
    logging.info("Aggregated to %d CaSPAr rows and %d ERA5 rows", 
                len(daily_caspar), len(daily_era5))
    return daily_caspar, daily_era5


def plot_daily(daily_caspar, daily_era5, metric, fname):
    # Color mapping for different lead times
    cmap = {24: 'r', 72: 'orange', 168: 'b', 240: 'k'}
    
    # Create 4 separate plots
    locations = ['Lytton', 'Vancouver']
    metrics = ['daily_max', 'daily_mean']
    
    for loc in locations:
        for met in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Filter CaSPAr data for this location and metric
            sub_data_caspar = daily_caspar[(daily_caspar.location == loc) & 
                                         (daily_caspar.lead_h.isin(cmap.keys()))]
            
            # Filter ERA5 data for this location and metric
            sub_data_era5 = daily_era5[(daily_era5.location == loc)]
            
            # Plot CaSPAr forecasts for each lead time
            for lh in sorted(cmap.keys()):
                lead_data = sub_data_caspar[sub_data_caspar.lead_h == lh]
                if not lead_data.empty:
                    ax.plot(lead_data.date, lead_data[met], 
                           color=cmap[lh], marker='o', linewidth=2, markersize=4,
                           label=f'{lh//24}-day')
            
            # Plot ERA5 ground truth
            if not sub_data_era5.empty:
                ax.plot(sub_data_era5.date, sub_data_era5[met], 
                       color='green', marker='s', linewidth=3, markersize=6,
                       label='ERA5 (Ground Truth)', linestyle='--')
            
            # Create descriptive title
            metric_name = "Daily Maximum" if met == "daily_max" else "Daily Mean"
            title = f"CaSPAr {loc} {metric_name} 2 m Temperature Forecasts vs ERA5\n(Forecast Valid Dates: 2021-06-25 to 2021-07-05)"
            
            ax.set(xlabel='Forecast Valid Date', ylabel='Temperature (°C)', title=title)
            ax.set_ylim(bottom=None, top=40)  # Set maximum y-axis value to 40°C
            ax.grid(True, alpha=0.3, color='grey', linestyle='-', linewidth=0.5)  # Add transparent grey grids
            ax.legend(title='Lead Time', ncol=1, fontsize='small')
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show dates every 2 days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Create filename based on location and metric
            loc_lower = loc.lower()
            met_suffix = "max" if met == "daily_max" else "mean"
            out = PLOT_DIR / f"caspar_vs_era5_{loc_lower}_{met_suffix}.png"
            plt.savefig(out, dpi=150)
            plt.close(fig)
            logging.info("Saved plot: %s", out)


def plot_maps(first_file):
    init = pd.to_datetime(first_file.stem, format="%Y%m%d%H")
    with xr.open_dataset(first_file) as ds:
        var = pick_temp_var(ds)
        da = ds[var]
        if getattr(da,'units','').lower().startswith(('k','kelvin')):
            da = da - 273.15
        lat2d, lon2d = ds.lat.values, ds.lon.values
        for lh in LEAD_HOURS:
            valid = init + pd.Timedelta(hours=lh)
            if np.datetime64(valid) in ds.time.values:
                tmp = da.sel(time=valid).squeeze()
                fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([lon2d.min(),lon2d.max(),lat2d.min(),lat2d.max()])
                ax.add_feature(cfeature.COASTLINE)
                im = ax.contourf(lon2d, lat2d, tmp, levels=20, transform=ccrs.PlateCarree())
                plt.colorbar(im, ax=ax, label='Temperature (°C)')
                for name, loc in POINTS.items():
                    ax.plot(loc['lon'], loc['lat'], 'ro', transform=ccrs.Geodetic())
                    ax.text(loc['lon']+0.2, loc['lat']+0.2, name,
                            transform=ccrs.Geodetic(), fontsize=9,
                            bbox={'facecolor':'white','alpha':0.6})
                
                # Enhanced title with CaSPAr information
                ax.set_title(f"CaSPAr Temperature Forecast\n"
                           f"Initialization: {init:%Y-%m-%d %H:00} UTC | "
                           f"Lead Time: {lh}h | "
                           f"Valid: {valid:%Y-%m-%d %H:00} UTC")
                out = PLOT_DIR / f"caspar_temp_map_L{lh}h_first_forecast.png"
                plt.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logging.info("Saved map: %s", out)
            else:
                logging.warning("Lead %dh not in %s", lh, first_file.name)


def main():
    files = sorted(DATA_DIR.glob('*.nc'))
    if not files:
        logging.error("No .nc files in %s", DATA_DIR)
        return
    
    inspect_files(files)
    
    # Extract both CaSPAr and ERA5 data
    df_caspar = extract_timeseries(files)
    df_era5 = extract_era5_data()
    
    save_csv(df_caspar, df_era5)
    daily_caspar, daily_era5 = aggregate_daily(df_caspar, df_era5)
    
    plot_daily(daily_caspar, daily_era5, 'daily_mean', 'daily_mean.png')
    plot_maps(files[0])
    
    logging.info("Script completed successfully.")

if __name__ == '__main__':
    main()
