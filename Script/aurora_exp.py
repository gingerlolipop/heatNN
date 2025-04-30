#!/usr/bin/env python
# filepath: /home/jing007/scratch/heatNN/script/aurora_exp.py

import os
from pathlib import Path

import torch
import xarray as xr
import matplotlib.pyplot as plt

from aurora import Batch, Metadata, Aurora, rollout

def main():
    # Define the path where the data is stored
    data_path = Path("/home/jing007/scratch/heatNN/dataraw")
    
    # Open the datasets (assumes the files are already downloaded)
    static_ds = xr.open_dataset(data_path / "static.nc", engine="netcdf4")
    surf_ds   = xr.open_dataset(data_path / "2023-01-01-surface-level.nc", engine="netcdf4")
    atmos_ds  = xr.open_dataset(data_path / "2023-01-01-atmospheric.nc", engine="netcdf4")
    
    i = 1  # select the time index

    # Build the Batch as required by the model
    batch = Batch(
        surf_vars={
            # Use consecutive time steps (i-1 and i) and a new batch dimension
            "2t": torch.from_numpy(surf_ds["t2m"].values[[i - 1, i]][None]),
            "10u": torch.from_numpy(surf_ds["u10"].values[[i - 1, i]][None]),
            "10v": torch.from_numpy(surf_ds["v10"].values[[i - 1, i]][None]),
            "msl": torch.from_numpy(surf_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            # Static variables use only the first time slice
            "z": torch.from_numpy(static_ds["z"].values[0]),
            "slt": torch.from_numpy(static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
        },
        atmos_vars={
            # Atmospheric variables also use consecutive time steps
            "t": torch.from_numpy(atmos_ds["t"].values[[i - 1, i]][None]),
            "u": torch.from_numpy(atmos_ds["u"].values[[i - 1, i]][None]),
            "v": torch.from_numpy(atmos_ds["v"].values[[i - 1, i]][None]),
            "q": torch.from_numpy(atmos_ds["q"].values[[i - 1, i]][None]),
            "z": torch.from_numpy(atmos_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values),
            lon=torch.from_numpy(surf_ds.longitude.values),
            time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_ds.pressure_level.values),
        ),
    )

    # Load and prepare the Aurora model.
    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.eval()
    model = model.to("cuda")

    # Run the model and get predictions for two steps.
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
    model = model.to("cpu")

    # Plot predictions against ERA5 data; save the plot to a file.
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for idx in range(2):
        pred = preds[idx]
        ax[idx, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
        ax[idx, 0].set_ylabel(str(pred.metadata.time[0]))
        if idx == 0:
            ax[idx, 0].set_title("Aurora Prediction")
        ax[idx, 0].set_xticks([])
        ax[idx, 0].set_yticks([])

        # Use a different time slice from the ERA5 surface-level data for comparison.
        ax[idx, 1].imshow(surf_ds["t2m"].values[2 + idx] - 273.15, vmin=-50, vmax=50)
        if idx == 0:
            ax[idx, 1].set_title("ERA5")
        ax[idx, 1].set_xticks([])
        ax[idx, 1].set_yticks([])

    plt.tight_layout()
    plt.savefig("predictions.png")
    print("Modeling step completed. Predictions saved as predictions.png.")

if __name__ == "__main__":
    main()