#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_daily_max_lytton_van.py

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
    保持经度在 [0,360)，裁剪到给定经纬范围，并 pad/crop 使经度与纬度大小可整除 patch_size。
    """
    ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
    lon_min360, lon_max360 = lon_min % 360, lon_max % 360
    ds = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min360, lon_max360))
    # pad longitude
    W = ds.sizes['longitude']
    pad_lon = (patch_size - (W % patch_size)) % patch_size
    if pad_lon:
        ds = ds.pad(longitude=(0, pad_lon), mode='edge')
        lon0 = ds.longitude.values[:-pad_lon]
        dlon = lon0[1] - lon0[0]
        extra = lon0[-1] + dlon * np.arange(1, pad_lon+1)
        ds = ds.assign_coords(longitude=np.concatenate([lon0, extra]))
    # crop latitude
    H = ds.sizes['latitude']
    crop_h = H - (H % patch_size)
    if crop_h < H:
        ds = ds.isel(latitude=slice(0, crop_h))
    return ds


def main():
    # —— 配置 ——
    data_dir = Path('/home/jing007/scratch/heatNN/dataraw')
    # 定义两个站点的坐标
    stations = {
        'Lytton': {'lat': 50.2333, 'lon': -121.7667},
        'Vancouver': {'lat': 49.2827, 'lon': -123.1207}
    }
    # 评估日期（按日）：2021-06-25 … 2021-07-05
    start_day = np.datetime64('2021-06-25')
    end_day   = np.datetime64('2021-07-05')
    days = np.arange(start_day, end_day + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    # 可用 init 时刻偏移
    offsets = [np.timedelta64(h, 'h') for h in (0, 6, 12, 18)]
    lead_list = [1, 3, 7, 14]

    # —— 读取 & 预处理 ERA5 数据 ——
    static_ds = xr.open_dataset(data_dir / 'static.nc', engine='netcdf4')
    surf_ds = xr.open_mfdataset([
        data_dir/'2021-06-surface-level.nc',
        data_dir/'2021-07-surface-level.nc'
    ], combine='by_coords', engine='netcdf4')
    atmos_ds = xr.open_mfdataset([
        data_dir/'2021-06-atmospheric.nc',
        data_dir/'2021-07-atmospheric.nc'
    ], combine='by_coords', engine='netcdf4')

    # —— 裁剪 & pad ——
    # 这里以 Lytton 所在区域为中心扩展20度，保证 Vancouver 也在区域内
    center = stations['Lytton']
    model = Aurora(use_lora=False)
    patch_size = model.patch_size
    static_ds = preprocess(static_ds, center['lon']-20, center['lon']+20, center['lat']-20, center['lat']+20, patch_size)
    surf_ds   = preprocess(surf_ds,   center['lon']-20, center['lon']+20, center['lat']-20, center['lat']+20, patch_size)
    atmos_ds  = preprocess(atmos_ds,  center['lon']-20, center['lon']+20, center['lat']-20, center['lat']+20, patch_size)

    # —— 定位站点格点索引 ——
    lons = surf_ds.longitude.values
    lats = surf_ds.latitude.values
    station_idx = {}
    for name, coord in stations.items():
        lon_mod = coord['lon'] % 360
        i_lon = np.abs(lons - lon_mod).argmin()
        i_lat = np.abs(lats - coord['lat']).argmin()
        station_idx[name] = {'i_lat': i_lat, 'i_lon': i_lon}

    # —— 离线加载模型 ——
    os.environ['HF_HUB_OFFLINE'] = '1'
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval().to('cuda')

    # —— 计算 ERA5 观测的日最大和日平均温度 ——
    times = surf_ds.valid_time.values.astype('datetime64[h]')
    obs = {}
    for name, idx in station_idx.items():
        t2m = surf_ds['t2m'].isel(latitude=idx['i_lat'], longitude=idx['i_lon']).values - 273.15
        dt_idx = pd.to_datetime(times.astype('datetime64[s]'))
        ser = pd.Series(t2m, index=dt_idx)
        daily_max = ser.resample('1D').max().reindex(pd.to_datetime(days.astype('datetime64[s]'))).values
        daily_mean = ser.resample('1D').mean().reindex(pd.to_datetime(days.astype('datetime64[s]'))).values
        obs[name] = {'daily_max': daily_max, 'daily_mean': daily_mean}

    # —— 预测并提取日最大和日平均 —— 
    # 结构：results[lead][站点]['daily_max'/'daily_mean']
    results = {}
    for lead in lead_list:
        results[lead] = {}
        for name, idx in station_idx.items():
            daily_max_list = []
            daily_mean_list = []
            for day in days:
                vals = []
                for off in offsets:
                    init = day - np.timedelta64(lead, 'D') + off
                    valid = surf_ds.valid_time.values.astype('datetime64[s]')
                    idxs = np.where(valid == init)[0]
                    if len(idxs) == 0:
                        continue
                    i = int(idxs[0])
                    batch = Batch(
                        surf_vars={
                            '2t': torch.from_numpy(surf_ds['t2m'].values[[i-1, i]][None]),
                            '10u': torch.from_numpy(surf_ds['u10'].values[[i-1, i]][None]),
                            '10v': torch.from_numpy(surf_ds['v10'].values[[i-1, i]][None]),
                            'msl': torch.from_numpy(surf_ds['msl'].values[[i-1, i]][None]),
                        },
                        static_vars={
                            'z': torch.from_numpy(static_ds['z'].values[0]),
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
                            time=(pd.to_datetime(init).to_pydatetime(),),
                            atmos_levels=tuple(int(l) for l in atmos_ds.pressure_level.values),
                        ),
                    )
                    steps = lead * 4
                    with torch.inference_mode():
                        preds = [p.to('cpu') for p in rollout(model, batch.to('cuda'), steps=steps)]
                    pred_grid = preds[-1].surf_vars['2t'][0, 0].numpy() - 273.15
                    vals.append(pred_grid[idx['i_lat'], idx['i_lon']])
                if vals:
                    daily_max_list.append(max(vals))
                    daily_mean_list.append(np.mean(vals))
                else:
                    daily_max_list.append(np.nan)
                    daily_mean_list.append(np.nan)
            results[lead][name] = {
                'daily_max': np.array(daily_max_list),
                'daily_mean': np.array(daily_mean_list)
            }
            # 保存中间预测结果 CSV
            df_pred = pd.DataFrame({
                'date': pd.to_datetime(days.astype('datetime64[s]')),
                'pred_daily_max': results[lead][name]['daily_max'],
                'pred_daily_mean': results[lead][name]['daily_mean']
            })
            out_pred_csv = f'{name}_daily_pred_{lead}d.csv'
            df_pred.to_csv(out_pred_csv, index=False)
            print(f'Saved intermediate predictions: {out_pred_csv}')

    # —— 绘图比较 ——
    cmap = {1:'r', 3:'orange', 7:'b', 14:'k'}
    # 1. Lytton Daily Max
    plt.figure(figsize=(12,6))
    plt.plot(days, obs['Lytton']['daily_max'], 'g-x', label='ERA5 Daily Max')
    for lead in lead_list:
        plt.plot(days, results[lead]['Lytton']['daily_max'], marker='o', color=cmap[lead],
                 label=f'{lead}-day lead')
    plt.title('Forecasted Daily Maximum 2m Temp at Lytton (BC)')
    plt.xlabel('Date')
    plt.ylabel('T2m Daily Max (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = 'forecast_daily_max_Lytton_all_leads.png'
    plt.savefig(out_png)
    print(f'Saved plot: {out_png}')
    
    # 2. Vancouver Daily Max
    plt.figure(figsize=(12,6))
    plt.plot(days, obs['Vancouver']['daily_max'], 'g-x', label='ERA5 Daily Max')
    for lead in lead_list:
        plt.plot(days, results[lead]['Vancouver']['daily_max'], marker='o', color=cmap[lead],
                 label=f'{lead}-day lead')
    plt.title('Forecasted Daily Maximum 2m Temp at Vancouver (BC)')
    plt.xlabel('Date')
    plt.ylabel('T2m Daily Max (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = 'forecast_daily_max_Vancouver_all_leads.png'
    plt.savefig(out_png)
    print(f'Saved plot: {out_png}')
    
    # 3. Lytton Daily Mean
    plt.figure(figsize=(12,6))
    plt.plot(days, obs['Lytton']['daily_mean'], 'g-x', label='ERA5 Daily Mean')
    for lead in lead_list:
        plt.plot(days, results[lead]['Lytton']['daily_mean'], marker='o', color=cmap[lead],
                 label=f'{lead}-day lead')
    plt.title('Forecasted Daily Mean 2m Temp at Lytton (BC)')
    plt.xlabel('Date')
    plt.ylabel('T2m Daily Mean (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = 'forecast_daily_mean_Lytton_all_leads.png'
    plt.savefig(out_png)
    print(f'Saved plot: {out_png}')
    
    # 4. Vancouver Daily Mean
    plt.figure(figsize=(12,6))
    plt.plot(days, obs['Vancouver']['daily_mean'], 'g-x', label='ERA5 Daily Mean')
    for lead in lead_list:
        plt.plot(days, results[lead]['Vancouver']['daily_mean'], marker='o', color=cmap[lead],
                 label=f'{lead}-day lead')
    plt.title('Forecasted Daily Mean 2m Temp at Vancouver (BC)')
    plt.xlabel('Date')
    plt.ylabel('T2m Daily Mean (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = 'forecast_daily_mean_Vancouver_all_leads.png'
    plt.savefig(out_png)
    print(f'Saved plot: {out_png}')


if __name__ == '__main__':
    main()