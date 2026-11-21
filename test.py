import pandas as pd
steam_data = pd.read_csv("data/steam_app_data.csv", nrows=5)
print(steam_data.columns.tolist())

steam_data = pd.read_csv(
    "data/steam_app_data.csv",
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
    "data/steam_app_data.csv",
    usecols=["steam_appid", "genres"]
)

duplicates = metadata[
    metadata["steam_appid"].duplicated(keep=False)
]

print("Duplicate metadata records:", len(duplicates))
print("Affected app IDs:", duplicates["steam_appid"].nunique())