from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEATURE_PATH = "data/features/features_raw_NW-88010013_ALLFULL_20260216T055951Z_lb30_FEAT_V1.parquet"
OUT_DIR = Path("reports/phase3/seasonality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(FEATURE_PATH)

# Time features
df["phenomenonTime"] = pd.to_datetime(df["phenomenonTime"], errors="coerce")
df["year"] = df["phenomenonTime"].dt.year
df["month"] = df["phenomenonTime"].dt.month

# Recompute y if needed
df["y"] = (df["chl_ugL"] > 20.0).astype(int)

# -------------------------
# 1️⃣ Monthly Mean Chlorophyll
# -------------------------
monthly_mean = df.groupby("month")["chl_ugL"].mean()

plt.figure()
plt.plot(monthly_mean.index, monthly_mean.values, marker="o")
plt.title("Monthly Mean Chlorophyll-a")
plt.xlabel("Month")
plt.ylabel("Mean Chlorophyll (µg/L)")
plt.tight_layout()
plt.savefig(OUT_DIR / "monthly_mean_chl.png", dpi=200)
plt.close()

# -------------------------
# 2️⃣ Monthly Exceedance Rate
# -------------------------
monthly_exceed = df.groupby("month")["y"].mean()

plt.figure()
plt.plot(monthly_exceed.index, monthly_exceed.values, marker="o")
plt.title("Monthly Exceedance Rate (Chl > 20 µg/L)")
plt.xlabel("Month")
plt.ylabel("Exceedance Probability")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR / "monthly_exceedance_rate.png", dpi=200)
plt.close()

# -------------------------
# 3️⃣ Year-Month Heatmap (Mean Chlorophyll)
# -------------------------
pivot = df.pivot_table(index="year", columns="month", values="chl_ugL", aggfunc="mean")

plt.figure()
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="Mean Chlorophyll (µg/L)")
plt.xticks(range(12), range(1, 13))
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Year–Month Mean Chlorophyll Heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "year_month_heatmap.png", dpi=200)
plt.close()

# -------------------------
# 4️⃣ Seasonal Distribution
# -------------------------
def month_to_season(m):
    if m in [12, 1, 2]:
        return "DJF"
    elif m in [3, 4, 5]:
        return "MAM"
    elif m in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"

df["season"] = df["month"].apply(month_to_season)

plt.figure()
df.boxplot(column="chl_ugL", by="season")
plt.yscale("log")
plt.title("Chlorophyll Distribution by Season")
plt.suptitle("")
plt.ylabel("Chlorophyll (µg/L)")
plt.tight_layout()
plt.savefig(OUT_DIR / "season_boxplot.png", dpi=200)
plt.close()

print("Saved seasonality plots to:", OUT_DIR)
