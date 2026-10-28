from pathlib import Path
from ast import literal_eval
import pandas as pd
pd.options.display.float_format = "{:,.2f}".format

data_dir = Path("data")
csv_file = data_dir / "all_data.csv"

if not csv_file.is_file():
    raise FileNotFoundError(
        f"Expected dataset file was not found: {csv_file}. "
        "Run dataloader.py first."
    )
# Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

# joining steam_app_data.csv
release_data = pd.read_csv(
    data_dir / "steam_app_data.csv",
    usecols=["steam_appid", "release_date"],
)
def extract_release_date(value):
    try:
        return literal_eval(value).get("date")
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return pd.NA

release_data["release_date"] = release_data["release_date"].map(
    extract_release_date
)
release_data = (
    release_data
    .rename(columns={"steam_appid": "appid"})
    .drop_duplicates("appid")
)
df = df.merge(
    release_data,
    on="appid",
    how="left",
    validate="many_to_one",
)


print(f"Original dataset: {len(df):,} games")
# drop dup games
df = df.drop_duplicates(subset="appid").copy()
# drop no name games
df = df.dropna(subset=["name"])
# remove any without reviews 
df = df[df["name"].str.strip() != ""].copy()
df = df[
    (df["positive"] >= 0) &
    (df["negative"] >= 0)
].copy()
print(f"After cleaning: {len(df):,} games")


# calc just using review statistic
# make a new col with total reviews for each games
df["total_reviews"] = df["positive"] + df["negative"]
# make a new positive rate col but replace total review if 0
df["positive_rate"] = (
    df["positive"] /
    df["total_reviews"].replace(0, float("nan"))
) * 100

print("\nOwner range examples:")
print(df["owners"].value_counts().head(15))


owner_ranges = (
    df["owners"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.split(r"\s*(?:\.\.|-)\s*", n=1, expand=True)
)
df["owners_min"] = pd.to_numeric(owner_ranges[0], errors="coerce")
df["owners_max"] = pd.to_numeric(owner_ranges[1], errors="coerce")
# calc mean of owner range
df["owners_estimate"] = (df["owners_min"] + df["owners_max"]) / 2
invalid_owners = (
    (df["owners_min"] < 0) |
    (df["owners_max"] < df["owners_min"])
)
df.loc[invalid_owners, "owners_estimate"] = float("nan")
print("\nMissing ownership estimates:")
print(df["owners_estimate"].isna().sum())

# 5 num summary for total reviews
print("\n--- Total Review Distribution ---")
print(
    df["total_reviews"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n--- Positive Rate Distribution ---")
print(df["positive_rate"].describe())

# 5 num sum for owner estimate
print("\n--- Owner Estimate Distribution ---")
print(
    df["owners_estimate"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n--- Games With No Reviews ---")
print((df["total_reviews"] == 0).sum())

# parameters for filtering
min_reviews = 2900
min_rating = 80
max_owners = 50000
filtered_games = df[
    (df["total_reviews"] >= min_reviews) &
    (df["positive_rate"] >= min_rating) &
    (df["owners_estimate"] <= max_owners)
].copy()

# print(
#     filtered_games
#     .sort_values(by="positive_rate", ascending=False)[["name", "release_date"]]
#     .head(15)
# )

ranked_games = filtered_games.sort_values(
    by=["positive_rate", "total_reviews"],
    ascending=[False, False]
)

print("\n--- Baseline Filtering Results ---")
print(f"Total games filtering: {len(df):,}")
print(f"Games passing filters: {len(filtered_games):,}")
print(f"Percentage retained: {len(filtered_games) / len(df) * 100:.2f}%")

print("\n--- Top 20 Hidden Gem Candidates ---")
print(
    ranked_games[
        ["name", "release_date", "positive_rate",
         "total_reviews", "owners", "owners_estimate"]]
         .head(20).to_string(index=False)
)