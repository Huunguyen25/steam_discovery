from ast import literal_eval
from pathlib import Path

import pandas as pd


def extract_release_date(value):
    try:
        return literal_eval(value).get("date")
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return pd.NA

def extract_genre_info(value):
    try:
        entries = literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []

    if not isinstance(entries, list):
        return []

    genres = {}

    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") is not None:
            genre_id = str(entry["id"])
            description = entry.get("description", "")
            genres[genre_id] = description

    return [
        {"id": genre_id, "description": description}
        for genre_id, description in genres.items()
    ]


def merge_genre_info(values):
    merged = {}

    for entries in values:
        for genre in entries:
            genre_id = genre["id"]
            description = genre["description"]

            if genre_id not in merged:
                merged[genre_id] = description

            elif not merged[genre_id] and description:
                merged[genre_id] = description

    return [
        {"id": genre_id, "description": description}
        for genre_id, description in merged.items()
    ]


def first_non_null(values):
    """Return the first available value from duplicate metadata records."""
    values = values.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def load_and_preprocess_data(data_dir=None):
    """Load game data, combine Steam metadata, and clean analysis features."""
    data_dir = (
        Path(data_dir) if data_dir is not None
        else Path(__file__).resolve().parent.parent / "data"
    )
    csv_file = data_dir / "all_data.csv"

    if not csv_file.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_file}. "
            "Run dataloader.py first."
        )
    df = pd.read_csv(csv_file)

    print(f"Original dataset: {len(df):,} games")

    # Load and parse Steam metadata.
    release_data = pd.read_csv(
        data_dir / "steam_app_data.csv",
        usecols=["steam_appid", "release_date", "genres"],
    )

    release_data["release_date"] = release_data["release_date"].map(
        extract_release_date
    )
    release_data["genre_info"] = release_data["genres"].map(extract_genre_info)

    # Combine duplicate records without losing dates or genre information.
    release_data = release_data.groupby("steam_appid", as_index=False).agg(
        release_date=("release_date", first_non_null),
        genre_info=("genre_info", merge_genre_info),
    )

    # Keep genre IDs for calculations and descriptions for readability.
    release_data["genre_ids"] = release_data["genre_info"].map(
        lambda genres: [genre["id"] for genre in genres]
    )

    release_data["genre_names"] = release_data["genre_info"].map(
        lambda genres: [genre["description"] for genre in genres]
    )

    release_data = release_data.drop(columns=["genre_info"]).rename(
        columns={"steam_appid": "appid"}
    )

    print(
        f"Unique metadata records after aggregation: {len(release_data):,}"
    )

    # Merge metadata into the main dataset.
    df = df.merge(
        release_data,
        on="appid",
        how="left",
        validate="many_to_one",
    )

    # Replace missing genre information with empty lists
    for column in ["genre_ids", "genre_names"]:
        df[column] = df[column].apply(
            lambda value: value if isinstance(value, list) else []
        )

    print("\n--- Metadata Validation ---")
    print(
        "Duplicate app IDs after aggregation:",
        release_data["appid"].duplicated().sum(),
    )
    print("Games with genre information:", df["genre_ids"].map(len).gt(0).sum())
    print("Games without genre information:", df["genre_ids"].map(len).eq(0).sum())

    # Deduplicate the main game dataset, not the aggregated metadata.
    df = df.drop_duplicates(subset="appid").copy()

    # Remove missing or blank names
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].str.strip()
    df = df[df["name"] != ""].copy()

    # Validate review counts
    for column in ["positive", "negative"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["positive", "negative"])

    df = df[(df["positive"] >= 0) & (df["negative"] >= 0)].copy()

    # Calculate review statistics
    df["total_reviews"] = df["positive"] + df["negative"]

    df["positive_rate"] = (
        df["positive"] / df["total_reviews"].replace(0, float("nan")) * 100
    )

    # Convert ownership ranges into numeric values
    owner_ranges = (
        df["owners"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"\s*(\d+)\s*(?:\.\.|-)\s*(\d+)\s*")
    )

    df["owners_min"] = pd.to_numeric(owner_ranges[0], errors="coerce")
    df["owners_max"] = pd.to_numeric(owner_ranges[1], errors="coerce")

    df["owners_estimate"] = (df["owners_min"] + df["owners_max"]) / 2

    # Validate ownership ranges
    invalid_owners = (
        (df["owners_min"] < 0) | (df["owners_max"] < df["owners_min"])
    )
    
    df.loc[invalid_owners, "owners_estimate"] = float("nan")

    print(f"After cleaning: {len(df):,} games")

    print("\n--- Genre Extraction Results ---")
    print("Games with genre information:")
    print(df["genre_ids"].map(len).gt(0).sum())

    print("\nGames without genre information:")
    print(df["genre_ids"].map(len).eq(0).sum())

    print("\nMost common genres:")
    print(df.explode("genre_ids")["genre_ids"].value_counts().head(15))

    print("\nExample games and their genres:")
    print(
        df[["name", "genre_ids", "genre_names"]]
        .head(15)
        .to_string(index=False)
    )

    return df

