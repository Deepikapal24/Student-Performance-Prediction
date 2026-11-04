"""Download the public Students Performance in Exams CSV into data/."""
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/NadavKiani/Students-Performance-in-Exams/master/StudentsPerformance.csv"
DEST = Path(__file__).parent / "data" / "StudentsPerformance.csv"
DEST.parent.mkdir(parents=True, exist_ok=True)
print("Downloading dataset...")
urllib.request.urlretrieve(URL, DEST)
print(f"Saved dataset to {DEST}")
