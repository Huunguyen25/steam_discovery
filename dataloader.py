from pathlib import Path
import kagglehub

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# Download and extract the dataset's CSV files into data/.
dataset_path = Path(
    kagglehub.dataset_download(
        "fmpugliese/steam-all-games-data",
        output_dir=str(data_dir),
    )
)

print("Dataset downloaded to:", dataset_path)

csv_files = sorted(dataset_path.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset.")

print("CSV files downloaded:")
for csv_file in csv_files:
    print(f"- {csv_file}")
