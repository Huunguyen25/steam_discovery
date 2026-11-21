# Steam Hidden Gem Discovery

Find Steam games that have strong player reviews but relatively few owners. This project compares simple filtering, K-means clustering, and a custom score to identify possible hidden gems.

## Data

The project uses the Kaggle dataset `fmpugliese/steam-all-games-data`. The analysis requires two files in [data/](./data/):

- [all_data.csv](./data/all_data.csv): game names, review counts, and ownership ranges.
- [steam_app_data.csv](./data/steam_app_data.csv): release dates and genres.

[preprocessing.py](./Sources/preprocessing.py) joins these files by Steam app ID, combines duplicate metadata, removes duplicate games and invalid names or review counts, and extracts genres. It also calculates:

- **Total reviews:** positive reviews + negative reviews.
- **Positive rate:** the percentage of reviews that are positive.
- **Estimated owners:** the midpoint of the reported ownership range.

Games with missing information are excluded from methods that require that information.

## Project structure

```text
steam_discovery/
├── Sources/
│   ├── dataloader.py
│   ├── preprocessing.py
│   ├── discover.py
│   ├── kmeans.py
│   ├── scoring.py
│   ├── compare.py
│   ├── visualize.py
│   ├── main.py
│   ├── test.py
│   └── requirements.txt
├── data/
├── results/
│   ├── figures/
│   └── experiment CSV files and cluster plot
└── README.md
```

[Sources/](./Sources/) contains the code and dependencies. [data/](./data/) holds the downloaded dataset, and [results/](./results/) holds experiment outputs. Data and results stay at the project root when scripts run from inside [Sources/](./Sources/).

## Environment

The requested environment uses these exact versions. Package versions are pinned in [requirements.txt](./Sources/requirements.txt).

| Software | Version |
|---|---|
| Python | 3.14.4 |
| NumPy | 2.5.4 |
| Pandas | 3.1.0 |
| Scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| KaggleHub | 1.0.3 |

## Setup and run

Install Python **3.14.4** first. From the project root, create and activate a virtual environment, then enter [Sources/](./Sources/):

```bash
python3 -m venv env
source env/bin/activate
cd Sources
```

Check that `python3 --version` reports `3.14.4`. The commands below assume a Linux or macOS shell and an active virtual environment.

Run this sequence **from inside [Sources/](./Sources/)**:

```bash
pip install -r requirements.txt
python3 dataloader.py
python3 compare.py
python3 visualize.py
```

1. **Install:** installs the pinned packages into the active environment. Installation requires internet access and those versions to be available from your package index.
2. **Download:** downloads the Kaggle dataset into the root [data/](./data/) folder. This requires internet access. Skip this command if both required CSV files are already present. If downloaded into a nested folder, move the two required CSV files directly into [data/](./data/).
3. **Experiment:** cleans the data, runs all three methods, prints the comparison, and saves results. K-means displays a cluster plot; close its window to let the script continue.
4. **Visualize:** creates charts for cluster sizes, the top 10 scored games, method comparison, and scoring weight sensitivity. Close each plot window to continue to the next chart. Charts are saved in [results/figures/](./results/figures/).

To save charts without opening windows, use these alternatives from [Sources/](./Sources/):

```bash
MPLBACKEND=Agg python3 compare.py
MPLBACKEND=Agg python3 visualize.py
```

[main.py](./Sources/main.py) also runs all three methods, but use [compare.py](./Sources/compare.py) to save the comparison CSV files needed by the charts. [test.py](./Sources/test.py) is a helper for inspecting dataset columns, genres, categories, and duplicate metadata; it is not an automated test suite.

## Methods

### Basic filtering

[discover.py](./Sources/discover.py) selects games with:

- At least **2,900 reviews**.
- At least **80% positive reviews**.
- No more than **50,000 estimated owners**.

Qualifying games are ranked by positive rate, then review count, both highest first.

### K-means clustering

[kmeans.py](./Sources/kmeans.py) groups games by positive rate, total reviews, and estimated owners. It compresses large counts with a log transformation and standardizes the features so their different scales do not dominate the grouping.

It tries **2–8 clusters** and chooses the highest silhouette score, which measures how well the groups are separated. Within each cluster, it displays highly rated candidates with at least **100 reviews** and **90% positive reviews**.

PCA reduces the three features to two coordinates for the cluster chart. The clustering itself uses all three features.

### Custom scoring

[scoring.py](./Sources/scoring.py) combines two values:

- **Wilson score:** a cautious estimate of positive review share that accounts for review count.
- **Genre obscurity:** how low a game's estimated ownership ranks within its genres, averaged across its genres. Higher values mean fewer owners relative to peers.

```text
hidden_gem_score = 0.70 × wilson_score + 0.30 × genre_obscurity
```

Games are ranked from highest score to lowest. Scores are stored on a **0–1 scale**; console tables and the top-games chart display them on a **0–100 scale**.

The project also tests **80/20**, **70/30**, and **60/40** weights and measures how many top-20 games overlap with the 70/30 ranking.

## Comparison and saved results

[compare.py](./Sources/compare.py) compares the same number of top candidates from basic filtering and custom scoring: the smaller available list, capped at **20 games**. It reports average positive rate, median review count, median estimated owners, and overlap between the lists. K-means provides a separate view of game groups and the clusters containing the custom top 20.

The saved [comparison summary](./results/comparison_summary.csv) currently contains:

| Method | Games compared | Average positive rate | Median reviews | Median estimated owners |
|---|---:|---:|---:|---:|
| Basic Filtering | 16 | 92.43% | 3,800 | 35,000 |
| Custom Scoring | 16 | 99.38% | 302 | 10,000 |

In this saved run, custom scoring selected games with higher positive rates and fewer reviews and estimated owners. These results describe the selected lists; they do not prove that one method finds better games. Rerunning with different data or settings may change the results.

### Output files

| File | Contents |
|---|---|
| [baseline_results.csv](./results/baseline_results.csv) | All games that pass basic filtering, ranked |
| [kmeans_results.csv](./results/kmeans_results.csv) | Games with their cluster labels |
| [scoring_results.csv](./results/scoring_results.csv) | Games ranked by custom score |
| [comparison_summary.csv](./results/comparison_summary.csv) | Statistics for the equal-sized comparison |
| [comparison_top20.csv](./results/comparison_top20.csv) | Up to 20 custom-scored games with cluster labels |
| [weight_sensitivity.csv](./results/weight_sensitivity.csv) | Top-20 overlap for each weight setting |
| [kmeans_clusters.png](./results/kmeans_clusters.png) | PCA view of the K-means clusters |
| [figures/](./results/figures/) | Charts generated by the visualization script |

The method comparison chart reads the saved summary directly, so it uses the same statistics as the comparison script.

## Limitations

“Hidden gem” is defined here through reviews and estimated ownership, rather than a verified quality label. Ownership values are estimates, and missing genres or reviews can exclude games. Results also depend on the filtering thresholds, scoring weights, and dataset snapshot. The custom score adjusts for genre but does not account for a game's age or price.
