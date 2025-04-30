#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_exp_bc.py

import os
from pathlib import Path

# —— Matplotlib 配置，放在最前面 ——
cache_dir = Path('/scratch/st-tlwang-1/jing/matplotlib')
os.environ['MPLCONFIGDIR'] = str(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aurora import Batch, Metadata, Aurora, rollout

def main():
    # 数据路径
    data_path = Path('/home/jing007/scratch/heatNN/dataraw/era5_aurora_data')

    # 读 ERA5 全局数据集
    static_ds = xr.open_dataset(data_path / 'static.nc', engine='netcdf4')
    surf_ds   = xr.open_dataset(data_path / '2023-01-01-surface-level.nc', engine='netcdf4')
    atmos_ds  = xr.open_dataset(data_path / '2023-01-01-atmospheric.nc', engine='netcdf4')

    # —— 裁剪到 British Columbia 区域 ——  
    # 原始 lat/lon 范围（deg）
    lat_min, lat_max = 49.0, 60.0
    lon_min, lon_max = -139.0, -114.0

    # 映射到 [0,360) 区间
    lon_min360 = lon_min % 360   # -139 -> 221
    lon_max360 = lon_max % 360   # -114 -> 246

    # 针对每个数据集：先把 longitude mod 360，再升序排序，最后裁剪
    def crop_ds(ds):
        return (
            ds
            .assign_coords(longitude=(ds.longitude % 360))
            .sortby('longitude')
            .sel(
                latitude = slice(lat_max, lat_min),
                longitude = slice(lon_min360, lon_max360),
            )
        )

    static_ds = crop_ds(static_ds)
    surf_ds   = crop_ds(surf_ds)
    atmos_ds  = crop_ds(atmos_ds)

    # 选择单时间索引 (i-1, i) 做一次推断
    i = 1

    # 构造 Batch（各张量 shape={batch=1, steps=2, lat, lon}）
    batch = Batch(
        surf_vars = {
            '2t':  torch.from_numpy(surf_ds['t2m'].values[[i-1, i]][None]),
            '10u': torch.from_numpy(surf_ds['u10'].values[[i-1, i]][None]),
            '10v': torch.from_numpy(surf_ds['v10'].values[[i-1, i]][None]),
            'msl': torch.from_numpy(surf_ds['msl'].values[[i-1, i]][None]),
        },
        static_vars = {
            'z':   torch.from_numpy(static_ds['z'].values[0]),
            'slt': torch.from_numpy(static_ds['slt'].values[0]),
            'lsm': torch.from_numpy(static_ds['lsm'].values[0]),
        },
        atmos_vars = {
            't': torch.from_numpy(atmos_ds['t'].values[[i-1, i]][None]),
            'u': torch.from_numpy(atmos_ds['u'].values[[i-1, i]][None]),
            'v': torch.from_numpy(atmos_ds['v'].values[[i-1, i]][None]),
            'q': torch.from_numpy(atmos_ds['q'].values[[i-1, i]][None]),
            'z': torch.from_numpy(atmos_ds['z'].values[[i-1, i]][None]),
        },
        metadata = Metadata(
            lat = torch.from_numpy(surf_ds.latitude.values),
            lon = torch.from_numpy(surf_ds.longitude.values),
            time = (surf_ds.valid_time.values.astype('datetime64[s]').tolist()[i],),
            atmos_levels = tuple(int(l) for l in atmos_ds.pressure_level.values),
        ),
    )

    # —— 离线加载 & GPU 设置 ——  
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model = Aurora(use_lora=False)
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval().to('cuda')

    # 推断两步
    with torch.inference_mode():
        preds = [pred.to('cpu') for pred in rollout(model, batch, steps=2)]

    # —— 画图并保存 ——  
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for idx in range(2):
        pred = preds[idx]
        # Aurora 预报
        ax[idx, 0].imshow(pred.surf_vars['2t'][0, 0].numpy() - 273.15,
                          vmin=-50, vmax=50)
        ax[idx, 0].set_ylabel(str(pred.metadata.time[0]))
        if idx == 0:
            ax[idx, 0].set_title('Aurora Prediction (BC)')
        ax[idx, 0].axis('off')

        # 对比 ERA5
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
