from preprocessing import load_and_preprocess_data

def run(df):
    # Baseline filtering parameters
    min_reviews = 2900
    min_rating = 80
    max_owners = 50000

    # Apply filtering
    filtered_games = df[
        (df["total_reviews"] >= min_reviews) &
        (df["positive_rate"] >= min_rating) &
        (df["owners_estimate"] <= max_owners)
    ].copy()

    # Rank qualifying games
    ranked_games = filtered_games.sort_values(
        by=["positive_rate", "total_reviews"],
        ascending=[False, False]
    )

    print("\n--- Baseline Filtering Results ---")
    print(f"Total games: {len(df):,}")
    print(f"Games passing filters: {len(filtered_games):,}")
    print(
        f"Percentage retained: "
        f"{len(filtered_games) / len(df) * 100:.2f}%"
    )

    print("\n--- Baseline Candidates ---")

    print(
        ranked_games[
            ["name", "positive_rate", "total_reviews",
             "owners_estimate"]
        ].head(20).to_string(index=False)
    )

    return ranked_games

if __name__ == "__main__":
    df = load_and_preprocess_data()
    run(df)
