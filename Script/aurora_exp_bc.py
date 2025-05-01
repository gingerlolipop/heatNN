#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_exp_bc.py

import os
from pathlib import Path

# —— Matplotlib 配置，避免只读主目录报错 ——
cache_dir = Path('/scratch/st-tlwang-1/jing/matplotlib')
os.environ['MPLCONFIGDIR'] = str(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

import torch
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aurora import Batch, Metadata, Aurora, rollout

def main():
    # —— 初始化 ——
    data_path = Path('/home/jing007/scratch/heatNN/dataraw/era5_aurora_data')
    # 获取 patch_size，用于后续 pad 操作
    patch_size = Aurora(use_lora=False).patch_size

    # —— 读取 ERA5 全局数据 ——
    static_ds = xr.open_dataset(data_path / 'static.nc', engine='netcdf4')
    surf_ds   = xr.open_dataset(data_path / '2023-01-01-surface-level.nc', engine='netcdf4')
    atmos_ds  = xr.open_dataset(data_path / '2023-01-01-atmospheric.nc', engine='netcdf4')

    # BC 区域经纬度（deg）
    lat_min, lat_max = 49.0, 60.0
    lon_min, lon_max = -139.0, -114.0
    # 映射到 [0,360)
    lon_min360 = lon_min % 360
    lon_max360 = lon_max % 360

    def preprocess(ds):
        # 保持 [0,360) 并升序
        ds = ds.assign_coords(longitude=(ds.longitude % 360)) \
               .sortby('longitude')
        # 裁剪到 BC
        ds = ds.sel(latitude=slice(lat_max, lat_min),
                    longitude=slice(lon_min360, lon_max360))
        # pad 到 patch_size 的倍数
        w = ds.sizes['longitude']
        pad = (patch_size - (w % patch_size)) % patch_size
        if pad > 0:
            ds = ds.pad(longitude=(0, pad), mode='edge')
            # 重建严格递增坐标
            orig = ds.longitude.values[:-pad]
            res = orig[1] - orig[0]
            extra = orig[-1] + res * np.arange(1, pad+1)
            ds = ds.assign_coords(longitude=np.concatenate([orig, extra]))
        return ds

    static_ds = preprocess(static_ds)
    surf_ds   = preprocess(surf_ds)
    atmos_ds  = preprocess(atmos_ds)

    # 单步推断索引
    i = 1

    # —— 构建 Batch ——
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
            time=(surf_ds.valid_time.values.astype('datetime64[s]').tolist()[i],),
            atmos_levels=tuple(int(l) for l in atmos_ds.pressure_level.values),
        ),
    )

    # —— 离线加载 & GPU 准备 ——
    os.environ['HF_HUB_OFFLINE'] = '1'
    model = Aurora(use_lora=False)
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval()
    # 单卡模式：直接搬至 GPU
    model.to('cuda')

    # —— 推断并搬到 CPU ——
    with torch.inference_mode():
        preds = [pred.to('cpu') for pred in rollout(model, batch.to('cuda'), steps=2)]

    # —— 可视化 & 保存 ——
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for idx in range(2):
        p = preds[idx]
        ax[idx, 0].imshow(p.surf_vars['2t'][0, 0].numpy() - 273.15,
                          vmin=-50, vmax=50)
        ax[idx, 0].set_ylabel(str(p.metadata.time[0]))
        if idx == 0:
            ax[idx, 0].set_title('Aurora Prediction (BC)')
        ax[idx, 0].axis('off')

        ax[idx, 1].imshow(surf_ds['t2m'].values[2 + idx] - 273.15,
                          vmin=-50, vmax=50)
        if idx == 0:
            ax[idx, 1].set_title('ERA5 Ground Truth (BC)')
        ax[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_bc.png')
    print('Modeling step completed. Predictions saved as predictions_bc.png.')

if __name__ == '__main__':
    main()
