
from pathlib import Path
import pandas as pd
from preprocessing import load_and_preprocess_data
from discover import run as run_baseline
from kmeans import run as run_kmeans
from scoring import run as run_scoring


def summarize_results(name, games):

    return {
        "method": name,
        "game_count": len(games),
        "avg_positive_rate": games["positive_rate"].mean(),
        "median_reviews": games["total_reviews"].median(),
        "median_owners": games["owners_estimate"].median()
    }


def run_comparison(df):

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ----------------------------------
    # 1. Basic Filtering
    # ----------------------------------

    print("\n========== BASIC FILTERING ==========")
    baseline_games = run_baseline(df)

    # ----------------------------------
    # 2. K-Means
    # ----------------------------------
    print("\n========== K-MEANS ==========")
    clustered_games = run_kmeans(df)

    # ----------------------------------
    # 3. Custom Scoring
    # ----------------------------------

    print("\n========== CUSTOM SCORING ==========")

    scored_games = run_scoring(df)

    scored_top20 = scored_games.head(20).copy()

    # ----------------------------------
    # Fair comparison using equal N
    # ----------------------------------

    print("\n========== METHOD COMPARISON ==========")

    comparison_size = min(
        len(baseline_games),
        len(scored_games),
        20
    )

    baseline_top = baseline_games.head(comparison_size)
    scoring_top = scored_games.head(comparison_size)

    summary = pd.DataFrame([
        summarize_results(
            "Basic Filtering",
            baseline_top
        ),
        summarize_results(
            "Custom Scoring",
            scoring_top
        )
    ])

    print("\n--- Equal-Sized Ranking Comparison ---")
    print(summary.round(2).to_string(index=False))

    # ----------------------------------
    # Candidate overlap
    # ----------------------------------

    baseline_ids = set(baseline_top["appid"])
    scoring_ids = set(scoring_top["appid"])

    overlap = baseline_ids.intersection(scoring_ids)
    
    shared_games = baseline_top[baseline_top["appid"].isin(overlap)]

    print("\n--- Shared Games ---")

    print(
        shared_games[
            ["appid", "name", "positive_rate", "total_reviews"]
        ].to_string(index=False)
    )
    union = baseline_ids.union(scoring_ids)

    print("\n--- Candidate Overlap ---")
    print(f"Comparison size: {comparison_size}")
    print(f"Shared games: {len(overlap)}")

    if union:
        print(f"Jaccard similarity: {len(overlap) / len(union):.2%}")
    else:
        print("Jaccard similarity: N/A")

    if baseline_ids:
        print(f"Baseline overlap: {len(overlap) / len(baseline_ids):.2%}")

    if scoring_ids:
        print(f"Scoring overlap: {len(overlap) / len(scoring_ids):.2%}")

    # ----------------------------------
    # 6. Analyze K-Means clusters
    # ----------------------------------

    print("\n--- K-Means Cluster Statistics ---")

    cluster_summary = (
        clustered_games
        .groupby("cluster")
        .agg(
            game_count=("appid", "count"),
            avg_positive_rate=("positive_rate", "mean"),
            median_reviews=("total_reviews", "median"),
            median_owners=("owners_estimate", "median")
        )
        .round(2)
    )

    print(cluster_summary.to_string())

    # ----------------------------------
    # 7. Identify clusters containing
    #    custom scoring candidates
    # ----------------------------------

    top20_with_cluster = scored_top20.merge(
        clustered_games[["appid", "cluster"]],
        on="appid",
        how="left",
        validate="one_to_one"
    )

    print("\n--- Custom Top 20 by K-Means Cluster ---")

    print(
        top20_with_cluster["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ----------------------------------
    # 8. Save results
    # ----------------------------------

    baseline_games.to_csv(
        results_dir / "baseline_results.csv",
        index=False
    )

    clustered_games.to_csv(
        results_dir / "kmeans_results.csv",
        index=False
    )

    scored_games.to_csv(
        results_dir / "scoring_results.csv",
        index=False
    )

    top20_with_cluster.to_csv(
        results_dir / "comparison_top20.csv",
        index=False
    )

    summary.to_csv(
        results_dir / "comparison_summary.csv",
        index=False
    )

    print("\nResults saved in the results/ directory.")

    return {
        "baseline": baseline_games,
        "clusters": clustered_games,
        "scoring": scored_games,
        "summary": summary,
        "top20": top20_with_cluster
    }


if __name__ == "__main__":

    df = load_and_preprocess_data()

    run_comparison(df)
