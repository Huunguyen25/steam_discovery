
from preprocessing import load_and_preprocess_data

from discover import run as run_baseline

# uncomment important and function calls later

from kmeans import run as run_kmeans
from scoring import run as run_scoring


def main():

    # Load and preprocess only once
    df = load_and_preprocess_data()

    # Method 1: Basic filtering
    baseline_results = run_baseline(df)

    # Method 2: K-means
    kmeans_results = run_kmeans(df)

    # Method 3: Custom scoring
    scoring_results = run_scoring(df)


if __name__ == "__main__":
    main()
