from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run(df):
    features = [
        "positive_rate",
        "total_reviews",
        "owners_estimate"
    ]
    # 1. Select clustering features
    cluster_data = df[["positive_rate", "total_reviews", "owners_estimate"]].copy()
    
    # Remove missing or invalid values
    cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Games available for clustering: {len(cluster_data):,}")


    cluster_data["total_reviews"] = np.log1p(cluster_data["total_reviews"])
    cluster_data["owners_estimate"] = np.log1p(cluster_data["owners_estimate"])

    # 3. Normalize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # 4. Experiment with different K values
    results = []
    print("\n--- K-Means Evaluation ---")

    for k in range(2, 9):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        # does the clustering and updating
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(
            scaled_data,
            labels,
            sample_size=10000,
            random_state=42
        )
        results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": score
        })
        print(
            f"K={k} | "
            f"Inertia={kmeans.inertia_:.2f} | "
            f"Silhouette={score:.4f}"
        )

    # 5. Select the best K
    best_k = max(
        results,
        key=lambda result: result["silhouette"]
    )["k"]

    print(f"\nBest K: {best_k}")

    # 6. Train final K-means model
    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )
    cluster_labels = final_kmeans.fit_predict(scaled_data)
    # Attach cluster labels to the original games
    clustered_games = df.loc[cluster_data.index].copy()
    clustered_games["cluster"] = cluster_labels

    # 7. Analyze cluster characteristics
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

    print("\n--- Cluster Summary ---")
    print(cluster_summary.to_string())


    # 8. Examine promising games within each cluster

    print("\n--- Potential Hidden Gems by Cluster ---")

    for cluster_id in sorted(clustered_games["cluster"].unique()):

        cluster_subset = clustered_games[
            (clustered_games["cluster"] == cluster_id) &
            (clustered_games["total_reviews"] >= 100) &
            (clustered_games["positive_rate"] >= 90)
        ]

        ranked_candidates = cluster_subset.sort_values(
            by=["positive_rate", "total_reviews"],
            ascending=[False, False]
        )

        print(f"\nCluster {cluster_id}")
        print(f"Qualifying games: {len(ranked_candidates):,}")

        print(
            ranked_candidates[
                [
                    "name",
                    "positive_rate",
                    "total_reviews",
                    "owners_estimate"
                ]
            ].head(10).to_string(index=False)
        )
        
    # Reduce normalized features to 2 dimensions
    pca = PCA(n_components=2)

    reduced_data = pca.fit_transform(scaled_data)

    # Plot clusters
    plt.figure(figsize=(10, 7))

    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.4,
        s=8
    )

    plt.title(f"Steam Games - K-Means Clustering (K={best_k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "kmeans_clusters.png", dpi=300)
    plt.show()


    return clustered_games

if __name__ == "__main__":
    df = load_and_preprocess_data()
    run(df)
