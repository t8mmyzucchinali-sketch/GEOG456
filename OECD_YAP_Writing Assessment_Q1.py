import pandas as pd
import numpy as np
import re

file_path = "Written_Test_YAP_DataAI copy.csv"

# -----------------------------
# 1) Load CSV
# -----------------------------
df = pd.read_csv(file_path, skiprows=3).dropna(axis=1, how="all")
df.columns = df.columns.str.strip()
df.rename(columns={df.columns[0]: "Country", df.columns[1]: "Year"}, inplace=True)

# Keep only expected columns
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
expected_cols = ["Country", "Year"] + months
df = df[expected_cols].copy()

# -----------------------------
# 2) Standardize country names
# -----------------------------
country_map = {
    "france": "France",
    "germany": "Germany",
    "brasil": "Brazil",
    "brazil": "Brazil",
    "china peoples rep": "China",
    "czech rep": "Czechia",
    "czech republic": "Czechia",
    "trkiye": "Turkiye",
    "turkey": "Turkiye",
    "uk": "United Kingdom",
    "usa": "United States",
    "us": "United States",
    "great britain": "United Kingdom",
    "ksa": "Saudi Arabia",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea rep": "South Korea",
}
country_map = {k.lower(): v for k, v in country_map.items()}

def canonical_country(name):
    if pd.isna(name):
        return np.nan
    cleaned = re.sub(r"[^A-Za-z\s]", "", str(name)).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return country_map.get(cleaned, cleaned.title())

df["Country"] = df["Country"].apply(canonical_country)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# -----------------------------
# 3) Clean monthly data
# -----------------------------
for m in months:
    df[m] = pd.to_numeric(df[m], errors="coerce")
    df.loc[df[m] < 0, m] = np.nan

# -----------------------------
# 4) Melt to long format
# -----------------------------
long = df.melt(id_vars=["Country", "Year"], value_vars=months,
               var_name="Month", value_name="GWh")

# -----------------------------
# 5) Detect and remove monthly outliers per country
# -----------------------------
outlier_mask = pd.Series(False, index=long.index)

for country, group in long.groupby("Country"):
    Q1 = group["GWh"].quantile(0.25)
    Q3 = group["GWh"].quantile(0.75)
    IQR = Q3 - Q1
    mask = (group["GWh"] < Q1 - 1.5*IQR) | (group["GWh"] > Q3 + 1.5*IQR)
    outlier_mask.loc[mask.index] = mask

# Replace outliers with NaN
long.loc[outlier_mask, "GWh"] = np.nan

# -----------------------------
# 6) Pivot back to monthly wide format
# -----------------------------
clean = long.pivot_table(index=["Country", "Year"], columns="Month", values="GWh").reset_index()
clean = clean[["Country", "Year"] + months]

# Compute annual generation
clean["Annual_Generation_GWh"] = clean[months].sum(axis=1, skipna=True)

# Save cleaned monthly data
clean.to_csv("cleaned_monthly_generation.csv", index=False)

# -----------------------------
# 7) Yearly summary table
# -----------------------------
yearly_avg_table = long.groupby(["Country", "Year"])["GWh"].sum().reset_index()
yearly_avg_table = yearly_avg_table.pivot(index="Country", columns="Year", values="GWh")

# Optional: add average annual generation
yearly_avg_table["Avg_Annual_Generation"] = yearly_avg_table.mean(axis=1)
yearly_avg_table = yearly_avg_table.reset_index()

yearly_avg_table.to_csv("yearly_avg_generation_by_country.csv", index=False)

print(yearly_avg_table)