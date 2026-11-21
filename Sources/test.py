from pathlib import Path

import pandas as pd

data_file = Path(__file__).resolve().parent.parent / "data" / "steam_app_data.csv"
steam_data = pd.read_csv(data_file, nrows=5)
print(steam_data.columns.tolist())

steam_data = pd.read_csv(
    data_file,
    usecols=["steam_appid", "genres", "categories"]
)

print("--- Example Genres ---")
print(
    steam_data["genres"]
    .dropna()
    .head(10)
    .to_string(index=False)
)

print("\n--- Example Categories ---")
print(
    steam_data["categories"]
    .dropna()
    .head(5)
    .to_string(index=False)
)

metadata = pd.read_csv(
    data_file,
    usecols=["steam_appid", "genres"]
)

duplicates = metadata[
    metadata["steam_appid"].duplicated(keep=False)
]

print("Duplicate metadata records:", len(duplicates))
print("Affected app IDs:", duplicates["steam_appid"].nunique())