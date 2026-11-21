from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project directories
results_dir = Path("results")
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

def plot_cluster_sizes():

    # Load K-means results
    df = pd.read_csv(
        results_dir / "kmeans_results.csv"
    )

    # Count games in each cluster
    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )

    # Add values above bars
    ax.bar_label(
        bars,
        labels=[f"{value:,}" for value in cluster_counts.values],
        padding=4
    )

    ax.set_title("K-Means Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Games")

    ax.set_ylim(0, cluster_counts.max() * 1.15)

    fig.tight_layout()

    # Save figure
    fig.savefig(
        figures_dir / "kmeans_cluster_distribution.png",
        dpi=300
    )

    plt.show()
    plt.close(fig)

def plot_top_hidden_gems():

    # Load custom scoring results
    df = pd.read_csv(
        results_dir / "scoring_results.csv"
    )

    # Select the top 10 games
    top_games = (
        df.nlargest(10, "hidden_gem_score")
        .copy()
    )

    # Convert decimal scores to a 0-100 scale if necessary
    if top_games["hidden_gem_score"].max() <= 1:
        top_games["hidden_gem_score"] *= 100

    # Sort bars by increasing score from left to right
    top_games = top_games.sort_values(
        "hidden_gem_score",
        ascending=True
    )

    # Create vertical bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        top_games["name"],
        top_games["hidden_gem_score"]
    )

    # Display scores above bars
    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=4
    )

    ax.set_title("Top 10 Hidden-Gem Candidates")
    ax.set_xlabel("Game")
    ax.set_ylabel("Hidden-Gem Score (out of 100)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_ylim(0, 105)

    fig.tight_layout()

    # Save chart
    fig.savefig(
        figures_dir / "top_10_hidden_gems.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def plot_method_comparison():

    # Load comparison statistics produced by compare.py
    summary = pd.read_csv(
        results_dir / "comparison_summary.csv"
    )

    methods = summary["method"].tolist()
    avg_ratings = summary["avg_positive_rate"].tolist()
    median_reviews = summary["median_reviews"].tolist()

    # Create two charts side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: Average positive rating
    bars1 = axes[0].bar(
        methods,
        avg_ratings
    )

    axes[0].bar_label(
        bars1,
        fmt="%.2f%%",
        padding=4
    )

    axes[0].set_title("Average Positive Rating")
    axes[0].set_ylabel("Positive Rate (%)")
    axes[0].set_ylim(0, 110)

    # Chart 2: Median review count
    bars2 = axes[1].bar(
        methods,
        median_reviews
    )

    axes[1].bar_label(
        bars2,
        fmt="%.0f",
        padding=4
    )

    axes[1].set_title("Median Review Count")
    axes[1].set_ylabel("Number of Reviews")
    axes[1].set_ylim(0, max(median_reviews) * 1.2)

    fig.suptitle(
        "Basic Filtering vs. Custom Scoring",
        fontsize=14
    )

    fig.tight_layout()

    # Save chart
    fig.savefig(
        figures_dir / "method_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def plot_weight_sensitivity():

    # Load custom scoring results
    df = pd.read_csv(
        results_dir / "scoring_results.csv"
    )

    # Number of candidates to compare
    top_n = 20

    # Different weight configurations
    configurations = {
        "80/20": (0.80, 0.20),
        "70/30": (0.70, 0.30),
        "60/40": (0.60, 0.40)
    }

    rankings = {}

    # Calculate rankings for each configuration
    for name, (wilson_weight, obscurity_weight) in configurations.items():

        df["test_score"] = (
            df["wilson_score"] * wilson_weight +
            df["genre_obscurity"] * obscurity_weight
        )

        ranked = df.sort_values(
            by=["test_score", "appid"],
            ascending=[False, True]
        )

        rankings[name] = set(
            ranked.head(top_n)["appid"]
        )

    # Use 70/30 as the reference ranking
    reference = rankings["70/30"]

    overlap_counts = []
    overlap_percentages = []

    for name in configurations:

        shared = len(
            reference.intersection(rankings[name])
        )

        percentage = shared / top_n * 100

        overlap_counts.append(shared)
        overlap_percentages.append(percentage)

        print(
            f"{name}: {shared}/{top_n} games "
            f"({percentage:.1f}% overlap)"
        )

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        list(configurations.keys()),
        overlap_percentages
    )

    # Display percentages
    ax.bar_label(
        bars,
        labels=[
            f"{value:.0f}%"
            for value in overlap_percentages
        ],
        padding=4
    )

    ax.set_title("Weight Sensitivity Analysis")
    ax.set_xlabel("Wilson / Genre Obscurity Weights")
    ax.set_ylabel("Overlap with 70/30 Top 20 (%)")

    ax.set_ylim(0, 110)

    fig.tight_layout()

    # Save chart
    fig.savefig(
        figures_dir / "weight_sensitivity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Save numerical results for the report
    sensitivity_results = pd.DataFrame({
        "configuration": list(configurations.keys()),
        "shared_games": overlap_counts,
        "overlap_percentage": overlap_percentages
    })

    sensitivity_results.to_csv(
        results_dir / "weight_sensitivity.csv",
        index=False
    )


if __name__ == "__main__":
    plot_cluster_sizes()
    plot_top_hidden_gems()
    plot_method_comparison()
    plot_weight_sensitivity()
