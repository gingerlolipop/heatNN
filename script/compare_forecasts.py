#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aurora import Batch, Metadata, Aurora, rollout
import logging

# Set up logging and environment variables
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def preprocess_aurora(ds, lon_min, lon_max, lat_min, lat_max, patch_size):
    """Preprocess dataset for Aurora model"""
    ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
    lon_min360, lon_max360 = lon_min % 360, lon_max % 360
    ds = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min360, lon_max360))
    
    # Pad longitude
    W = ds.sizes['longitude']
    pad_lon = (patch_size - (W % patch_size)) % patch_size
    if pad_lon:
        ds = ds.pad(longitude=(0, pad_lon), mode='edge')
        lon0 = ds.longitude.values[:-pad_lon]
        dlon = lon0[1] - lon0[0]
        extra = lon0[-1] + dlon * np.arange(1, pad_lon+1)
        ds = ds.assign_coords(longitude=np.concatenate([lon0, extra]))
    
    # Crop latitude
    H = ds.sizes['latitude']
    crop_h = H - (H % patch_size)
    if crop_h < H:
        ds = ds.isel(latitude=slice(0, crop_h))
    return ds

def regrid_caspar(ds_caspar, ds_aurora):
    """Regrid CaSPAr data to Aurora's grid using xarray interpolation"""
    return ds_caspar.interp(
        latitude=ds_aurora.latitude,
        longitude=ds_aurora.longitude,
        method='linear'
    )

def compute_weighted_rmse(forecast, truth, lats):
    """Compute latitude-weighted RMSE"""
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    
    squared_diff = (forecast - truth)**2
    weighted_mean = np.average(squared_diff.mean(axis=-1), 
                             weights=weights,
                             axis=-1)
    return np.sqrt(weighted_mean)

def main():
    # Paths
    base_dir = Path('/home/jing007/scratch/heatNN')
    data_dir = base_dir / 'dataraw'
    caspar_dir = data_dir / 'caspar'
    plot_dir = base_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Time period
    start_date = np.datetime64('2021-06-25')
    end_date = np.datetime64('2021-07-05')
    lead_days = [1, 3, 7, 10]
    
    # Load ERA5 data
    logging.info("Loading ERA5 data...")
    era5_ds = xr.open_mfdataset([
        data_dir/'2021-06-surface-level.nc',
        data_dir/'2021-07-surface-level.nc'
    ], combine='by_coords')
    era5_t2m = era5_ds['t2m'] - 273.15
    
    # Load and process Aurora predictions
    logging.info("Processing Aurora predictions...")
    model = Aurora(use_lora=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval().to('cuda')
    
    # Initialize storage for RMSEs
    rmse_results = {
        'lead_days': lead_days,
        'aurora_rmse': [],
        'caspar_rmse': []
    }
    
    # Load CaSPAr data
    logging.info("Loading CaSPAr data...")
    caspar_files = sorted(caspar_dir.glob('*.nc'))
    caspar_ds = xr.open_mfdataset(caspar_files)
    
    # Regrid CaSPAr to Aurora grid
    logging.info("Regridding CaSPAr data...")
    caspar_regridded = regrid_caspar(caspar_ds, era5_ds)
    
    # Process each lead time
    for lead in lead_days:
        logging.info(f"Processing {lead}-day lead time...")
        
        # Get Aurora prediction
        steps = lead * 4  # 6h steps
        aurora_preds = []
        
        # Process through time steps
        valid_times = pd.date_range(start_date, end_date, freq='6H')
        for valid_time in valid_times:
            init_time = valid_time - pd.Timedelta(days=lead)
            try:
                i = np.where(era5_ds.valid_time.values == np.datetime64(init_time))[0][0]
                
                batch = Batch(
                    surf_vars={
                        '2t': torch.from_numpy(era5_ds['t2m'].values[[i-1, i]][None]),
                        '10u': torch.from_numpy(era5_ds['u10'].values[[i-1, i]][None]),
                        '10v': torch.from_numpy(era5_ds['v10'].values[[i-1, i]][None]),
                        'msl': torch.from_numpy(era5_ds['msl'].values[[i-1, i]][None]),
                    },
                    static_vars={
                        'z': torch.from_numpy(era5_ds['z'].values[0]),
                        'slt': torch.from_numpy(era5_ds['slt'].values[0]),
                        'lsm': torch.from_numpy(era5_ds['lsm'].values[0]),
                    },
                    metadata=Metadata(
                        lat=torch.from_numpy(era5_ds.latitude.values),
                        lon=torch.from_numpy(era5_ds.longitude.values),
                        time=(init_time.to_pydatetime(),),
                        atmos_levels=tuple(int(l) for l in era5_ds.pressure_level.values),
                    ),
                )
                
                with torch.inference_mode():
                    pred = rollout(model, batch.to('cuda'), steps=steps)[-1]
                aurora_preds.append(pred.surf_vars['2t'][0, 0].cpu().numpy() - 273.15)
            
            except IndexError:
                logging.warning(f"Skipping prediction for {valid_time}")
                continue
        
        aurora_preds = np.stack(aurora_preds)
        
        # Get CaSPAr prediction for this lead time
        caspar_lead = caspar_regridded.sel(lead_time=lead*24)  # hours
        
        # Compute RMSEs
        aurora_rmse = compute_weighted_rmse(
            aurora_preds,
            era5_t2m.values,
            era5_ds.latitude.values
        )
        
        caspar_rmse = compute_weighted_rmse(
            caspar_lead.values,
            era5_t2m.values,
            era5_ds.latitude.values
        )
        
        rmse_results['aurora_rmse'].append(aurora_rmse)
        rmse_results['caspar_rmse'].append(caspar_rmse)
        
        logging.info(f"Lead {lead}d - Aurora RMSE: {aurora_rmse:.3f}, CaSPAr RMSE: {caspar_rmse:.3f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_results['lead_days'], rmse_results['aurora_rmse'], 
             'ro-', label='Aurora', markersize=8)
    plt.plot(rmse_results['lead_days'], rmse_results['caspar_rmse'], 
             'bo-', label='CaSPAr', markersize=8)
    
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('Latitude-weighted RMSE (°C)')
    plt.title('Forecast Performance vs ERA5\n2021 British Columbia Heatwave')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_plot = plot_dir / 'rmse_vs_era5.png'
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved RMSE comparison plot: {out_plot}")

if __name__ == '__main__':
    main()