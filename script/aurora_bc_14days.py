#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_bc_14days.py

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


def preprocess(ds, lon_min, lon_max, lat_min, lat_max, patch_size):
    """
    保持经度在 [0,360)，裁剪到给定经纬范围，并 pad 经度与 latitude pad/crop 以满足 patch_size
    """
    ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
    lon_min360, lon_max360 = lon_min % 360, lon_max % 360
    ds = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min360, lon_max360))
    # pad longitude to patch_size
    W = ds.sizes['longitude']
    pad_lon = (patch_size - (W % patch_size)) % patch_size
    if pad_lon:
        ds = ds.pad(longitude=(0, pad_lon), mode='edge')
        lon0 = ds.longitude.values[:-pad_lon]
        dlon = lon0[1] - lon0[0]
        extra_lon = lon0[-1] + dlon * np.arange(1, pad_lon+1)
        ds = ds.assign_coords(longitude=np.concatenate([lon0, extra_lon]))
    # crop latitude to multiple of patch_size
    H = ds.sizes['latitude']
    crop_lat = H - (H % patch_size)
    if crop_lat < H:
        ds = ds.isel(latitude=slice(0, crop_lat))
    return ds


def main():
    data_dir = Path('/home/jing007/scratch/heatNN/dataraw')
    lat_min, lat_max = 48.3, 60.0
    lon_min, lon_max = -139.06, -114.03
    lead_days = 14
    start_eval = np.datetime64('2021-06-25')
    end_eval   = np.datetime64('2021-07-05')
    target_dates = np.arange(start_eval, end_eval + np.timedelta64(1,'D'), dtype='datetime64[D]')
    init_dates   = target_dates - np.timedelta64(lead_days, 'D')

    static_ds = xr.open_dataset(data_dir / 'static.nc', engine='netcdf4')
    surf_ds   = xr.open_mfdataset(
        [data_dir / '2021-06-surface-level.nc', data_dir / '2021-07-surface-level.nc'],
        combine='by_coords', engine='netcdf4')
    atmos_ds  = xr.open_mfdataset(
        [data_dir / '2021-06-atmospheric.nc', data_dir / '2021-07-atmospheric.nc'],
        combine='by_coords', engine='netcdf4')

    model = Aurora(use_lora=False)
    patch_size = model.patch_size

    static_ds = preprocess(static_ds, lon_min, lon_max, lat_min, lat_max, patch_size)
    surf_ds   = preprocess(surf_ds,   lon_min, lon_max, lat_min, lat_max, patch_size)
    atmos_ds  = preprocess(atmos_ds,  lon_min, lon_max, lat_min, lat_max, patch_size)

    times = surf_ds.valid_time.values

    os.environ['HF_HUB_OFFLINE'] = '1'
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval().to('cuda')

    rmse_list, mae_list = [], []
    for init in init_dates:
        i = int(np.where(times == init)[0][0])
        init_py = pd.to_datetime(init).to_pydatetime()
        batch = Batch(
            surf_vars={
                '2t':  torch.from_numpy(surf_ds['t2m'].values[[i-1, i]][None]),
                '10u': torch.from_numpy(surf_ds['u10'].values[[i-1, i]][None]),
                '10v': torch.from_numpy(surf_ds['v10'].values[[i-1, i]][None]),
                'msl': torch.from_numpy(surf_ds['msl'].values[[i-1, i]][None]),
            },
            static_vars={
                'z':   torch.from_numpy(static_ds['z'].values[0]),
                'slt': torch.from_numpy(static_ds['slt'].values[0]),
                'lsm': torch.from_numpy(static_ds['lsm'].values[0]),
            },
            atmos_vars={
                't': torch.from_numpy(atmos_ds['t'].values[[i-1, i]][None]),
                'u': torch.from_numpy(atmos_ds['u'].values[[i-1, i]][None]),
                'v': torch.from_numpy(atmos_ds['v'].values[[i-1, i]][None]),
                'q': torch.from_numpy(atmos_ds['q'].values[[i-1, i]][None]),
                'z': torch.from_numpy(atmos_ds['z'].values[[i-1, i]][None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_ds.latitude.values),
                lon=torch.from_numpy(surf_ds.longitude.values),
                time=(init_py,),
                atmos_levels=tuple(int(l) for l in atmos_ds.pressure_level.values),
            ),
        )

        steps = lead_days * 4
        with torch.inference_mode():
            preds = [p.to('cpu') for p in rollout(model, batch.to('cuda'), steps=steps)]

        pred = preds[-1].surf_vars['2t'][0, 0].numpy() - 273.15
        target = init + np.timedelta64(lead_days, 'D')
        truth = surf_ds['t2m'].values[np.where(times == target)[0][0]] - 273.15

        # Align shapes pred vs truth
        h, w = pred.shape
        truth_c = truth[:h, :w]

        rmse = np.sqrt(((pred - truth_c) ** 2).mean())
        mae  = np.abs(pred - truth_c).mean()
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(f"Init {init_py:%Y-%m-%d} → Target {target.astype(str)}: "
              f"RMSE={rmse:.3f}°C, MAE={mae:.3f}°C")

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(target_dates.astype('datetime64[D]'), rmse_list, 'o-', label='RMSE')
    plt.plot(target_dates.astype('datetime64[D]'), mae_list,  's-', label='MAE')
    plt.title('14-day Lead Forecast Errors: 2m Temperature (BC Region)')
    plt.xlabel('Target Date')
    plt.ylabel('Error (°C)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lead_forecast_errors.png')
    print('Saved plot: lead_forecast_errors.png')

if __name__ == '__main__':
    main()
