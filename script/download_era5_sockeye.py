#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/era5_download.py

import os
from pathlib import Path
import cdsapi

def main():
    # —— 指定 .cdsapirc 配置路径 ——  
    os.environ['CDSAPI_RC'] = '/home/jing007/scratch/heatNN/.cdsapirc'

    base = Path('/home/jing007/scratch/heatNN/dataraw')
    base.mkdir(parents=True, exist_ok=True)

    # 初始化 CDS API 客户端，读取上面指定的 .cdsapirc
    c = cdsapi.Client()

    # 1. 下载 static.nc（仅需一次）
    static_file = base / 'static.nc'
    if not static_file.exists():
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential',
                    'land_sea_mask',
                    'soil_type',
                ],
                'year': '2021',
                'month': '06',
                'day': '11',
                'time': '00:00',
                'format': 'netcdf',
            },
            str(static_file)
        )
        print('Downloaded static.nc')

    # 2. 按月下载 surface-level & atmospheric 数据
    for year, month, days in [
        ('2021', '06', list(range(11, 31))),
        ('2021', '07', list(range(1, 6))),
    ]:
        day_strs = [f"{d:02d}" for d in days]

        # surface‐level
        sl_file = base / f"{year}-{month}-surface-level.nc"
        if not sl_file.exists():
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '2m_temperature',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        'mean_sea_level_pressure',
                    ],
                    'year': year,
                    'month': month,
                    'day': day_strs,
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'format': 'netcdf',
                },
                str(sl_file)
            )
            print(f'Downloaded {sl_file.name}')

        # atmospheric
        at_file = base / f"{year}-{month}-atmospheric.nc"
        if not at_file.exists():
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        'temperature',
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'specific_humidity',
                        'geopotential',
                    ],
                    'pressure_level': [
                        '50','100','150','200','250','300','400',
                        '500','600','700','850','925','1000'
                    ],
                    'year': year,
                    'month': month,
                    'day': day_strs,
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'format': 'netcdf',
                },
                str(at_file)
            )
            print(f'Downloaded {at_file.name}')

    print('All ERA5 files downloaded.')

if __name__ == '__main__':
    main()
