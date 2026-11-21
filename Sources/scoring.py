
import numpy as np
import pandas as pd

from preprocessing import load_and_preprocess_data


def calculate_wilson_score(df, z=1.96):

    n = df["total_reviews"].astype(float)
    safe_n = n.where(n > 0)

    p = df["positive"] / safe_n

    denominator = 1 + (z ** 2 / safe_n)

    lower_bound = (
        p
        + (z ** 2 / (2 * safe_n))
        - z * np.sqrt(
            (p * (1 - p) / safe_n)
            + (z ** 2 / (4 * safe_n ** 2))
        )
    ) / denominator

    return lower_bound



def calculate_genre_obscurity(df):

    # Select the required columns
    genre_data = df[
        ["appid", "owners_estimate", "genre_ids"]
    ].copy()

    # Convert each game's genre list into separate rows
    genre_data = genre_data.explode("genre_ids")

    # Remove games without valid genre or ownership information
    genre_data = genre_data.dropna(
        subset=["genre_ids", "owners_estimate"]
    )

    # Calculate ownership percentile within each genre
    genre_data["ownership_percentile"] = (
        genre_data.groupby("genre_ids")["owners_estimate"]
        .rank(method="average", pct=True)
    )

    # Convert popularity into obscurity
    genre_data["genre_obscurity"] = (
        1 - genre_data["ownership_percentile"]
    )

    # Average obscurity across all genres belonging to each game
    obscurity_scores = (
        genre_data.groupby("appid")["genre_obscurity"]
        .mean()
    )

    return obscurity_scores



def run(df):

    # Calculate review confidence
    df = df.copy()

    df["wilson_score"] = calculate_wilson_score(df)

    # Calculate genre-relative obscurity
    obscurity_scores = calculate_genre_obscurity(df)

    df["genre_obscurity"] = df["appid"].map(
        obscurity_scores
    )

    # Remove games without enough information to score
    scored_games = df.dropna(
        subset=["wilson_score", "genre_obscurity"]
    ).copy()

    # Experiment with different scoring weights
    print("\n--- Weight Comparison ---")

    weights = [0.80, 0.70, 0.60]
    top20_results = {}

    for weight in weights:
        obscurity_weight = 1 - weight

        scored_games["test_score"] = (
            weight * scored_games["wilson_score"]
            + obscurity_weight * scored_games["genre_obscurity"]
        )

        ranked = scored_games.sort_values(
            by="test_score",
            ascending=False
        )

        top20 = ranked.head(20)
        top20_results[weight] = set(top20["appid"])

        print(
            f"\nWilson: {weight:.0%} | "
            f"Obscurity: {obscurity_weight:.0%}"
        )

        print(
            top20[
                ["name", "wilson_score", "genre_obscurity", "test_score"]
            ].head(10).to_string(index=False)
        )

    # Compare overlap between experiments
    print("\n--- Top 20 Ranking Overlap ---")

    for weight in weights:
        print(
            f"{weight:.0%} Wilson vs 70% Wilson: "
            f"{len(top20_results[weight] & top20_results[0.70])}/20 games"
        )

    # Combine both components
    df["hidden_gem_score"] = (
        0.70 * df["wilson_score"]
        + 0.30 * df["genre_obscurity"]
    )

    scored_games["hidden_gem_score"] = (
        0.70 * scored_games["wilson_score"]
        + 0.30 * scored_games["genre_obscurity"]
    )

    # Rank games by hidden-gem score
    ranked_games = scored_games.sort_values(
        by="hidden_gem_score",
        ascending=False
    )

    print("\n--- Custom Scoring Results ---")

    print(f"Total games: {len(df):,}")
    print(f"Games successfully scored: {len(scored_games):,}")

    print("\n--- Top 20 Hidden Gem Candidates ---")

    results = ranked_games[
        [
            "name",
            "genre_names",
            "positive_rate",
            "total_reviews",
            "wilson_score",
            "owners_estimate",
            "genre_obscurity",
            "hidden_gem_score"
        ]
    ].head(20).copy()

    # Convert scores into percentages for readability
    for column in [
        "wilson_score",
        "genre_obscurity",
        "hidden_gem_score"
    ]:
        results[column] *= 100

    print(results.round(2).to_string(index=False))

    return ranked_games


if __name__ == "__main__":
    df = load_and_preprocess_data()
    run(df)
