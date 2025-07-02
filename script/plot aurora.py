from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define the base result folder (adjust if yours lives elsewhere)
# For example, under your Windows user folder:
base = Path.home() / "Documents" / "GitHub" / "heatNN" / "result"

# Or relative to the script file:
# base = Path(__file__).parent / ".." / "result"

# Automatically find your CSVs by globbing:
csvs = { 
    p.stem.replace("_metrics",""): p
    for p in base.glob("lead*d_metrics.csv")
}
# csvs now maps "lead1d" -> Path(.../lead1d_metrics.csv), etc.

# Define colors & zorder
colors = {"lead1d": "red", "lead3d": "orange", "lead7d": "blue", "lead14d": "black"}
zorders= {"lead1d": 1,       "lead3d": 2,        "lead7d": 3,       "lead14d": 4}

plt.figure(figsize=(12,6))
for name, path in csvs.items():
    df = pd.read_csv(path, parse_dates=["target_time"])
    plt.plot(
        df["target_time"], df["rmse"],
        label=name.replace("lead","").replace("d","-day"),
        color=colors.get(name,"gray"),
        zorder=zorders.get(name,0),
        marker="o", linewidth=2
    )

plt.xlabel("Target Time")
plt.ylabel("RMSE (°C)")
plt.title("RMSE of 2m Temperature Forecast by Lead Time")
plt.xticks(rotation=45)
plt.legend(title="Lead Time")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save back into the same folder
out = base / "rmse_all_leads.png"
plt.savefig(out)
plt.show()
print(f"Saved to {out}")
