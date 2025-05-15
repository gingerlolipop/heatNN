#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_bc_6h.py

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
    保持经度在 [0,360)，裁剪到区域，并 pad/crop 使经度和纬度大小可整除 patch_size。
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


def evaluate_lead(lead_days, surf_ds, atmos_ds, static_ds, model, times, target_times):
    """
    对给定的 lead_days（天）和一系列 target_times，按 6h 步长计算 RMSE/MAE。
    返回 DataFrame 包含 target_time, rmse, mae.
    """
    records = []
    for target in target_times:
        init = target - np.timedelta64(lead_days, 'D')
        # 索引
        i = int(np.where(times == init)[0][0])
        j = int(np.where(times == target)[0][0])
        # Python datetime for Metadata
        init_py = pd.to_datetime(init).to_pydatetime()
        # 构建 Batch
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
        # 多步预测
        steps = lead_days * 4  # 6h 一步
        with torch.inference_mode():
            preds = [p.to('cpu') for p in rollout(model, batch.to('cuda'), steps=steps)]
        # 取最后一步对应 target
        pred = preds[-1].surf_vars['2t'][0, 0].numpy() - 273.15
        truth = surf_ds['t2m'].values[j] - 273.15
        # 对齐形状
        h, w = pred.shape
        truth_c = truth[:h, :w]
        # 计算指标
        rmse = np.sqrt(((pred - truth_c)**2).mean())
        mae  = np.abs(pred - truth_c).mean()
        records.append({'target_time': target, 'rmse': rmse, 'mae': mae})
        print(f"Lead {lead_days}d | {target.astype('datetime64[h]')} UTC: RMSE={rmse:.3f}, MAE={mae:.3f}")
    return pd.DataFrame(records)


def main():
    # 配置
    data_dir = Path('/home/jing007/scratch/heatNN/dataraw')
    lat_min, lat_max = 48.3, 60.0
    lon_min, lon_max = -139.06, -114.03
    # 评估时间：2021-06-25 00:00 … 2021-07-05 18:00, 每 6h
    target_times = np.arange(
        np.datetime64('2021-06-25T00:00:00'),
        np.datetime64('2021-07-06T00:00:00'),  # 上限开区间
        np.timedelta64(6, 'h'),
        dtype='datetime64[s]'
    )
    # 加载数据
    static_ds = xr.open_dataset(data_dir / 'static.nc', engine='netcdf4')
    surf_ds = xr.open_mfdataset([
        data_dir/'2021-06-surface-level.nc', data_dir/'2021-07-surface-level.nc'
    ], combine='by_coords', engine='netcdf4')
    atmos_ds = xr.open_mfdataset([
        data_dir/'2021-06-atmospheric.nc', data_dir/'2021-07-atmospheric.nc'
    ], combine='by_coords', engine='netcdf4')
    # 模型 & patch_size
    os.environ['HF_HUB_OFFLINE'] = '1'
    model = Aurora(use_lora=False)
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval().to('cuda')
    patch_size = model.patch_size
    # 预处理
    static_ds = preprocess(static_ds, lon_min, lon_max, lat_min, lat_max, patch_size)
    surf_ds   = preprocess(surf_ds,   lon_min, lon_max, lat_min, lat_max, patch_size)
    atmos_ds  = preprocess(atmos_ds,  lon_min, lon_max, lat_min, lat_max, patch_size)
    times = surf_ds.valid_time.values

    # 对各 lead time 评估
    for lead in [1, 3, 7, 14]:
        df = evaluate_lead(lead, surf_ds, atmos_ds, static_ds, model, times, target_times)
        # 保存结果
        out_csv = f'lead{lead}d_metrics.csv'
        df.to_csv(out_csv, index=False)
        # 绘图
        plt.figure(figsize=(10,4))
        plt.plot(df['target_time'], df['rmse'], 'o-', label='RMSE')
        plt.plot(df['target_time'], df['mae'],  's-', label='MAE')
        plt.title(f'{lead}-Day Lead Forecast Errors (2m Temp, BC)')
        plt.xlabel('Target Time (UTC)')
        plt.ylabel('Error (°C)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        png = f'lead{lead}d_errors_6h.png'
        plt.savefig(png)
        print(f'Saved {out_csv} and {png}')

if __name__ == '__main__':
    main()
