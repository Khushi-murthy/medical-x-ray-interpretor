"""
Dataset Loader
Medical X-ray Interpreter
"""

from pathlib import Path
import pandas as pd

class DatasetLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.csv_path = self.dataset_path / "Data_Entry_2017.csv"

    def load_metadata(self):

        df = pd.read_csv(self.csv_path)

        print("=" * 60)
        print("Dataset Loaded Successfully")
        print("=" * 60)

        print(f"Total Images : {len(df)}")

        print(df.head())

        return df