import json
import pandas as pd
from pathlib import Path
from typing import Optional


class GenericDataset:
    """Generic dataset from a CSV file or a directory of JSON files.
    - Path is a file → load CSV; use text_column and optional category_column.
    - Path is a directory → load *.json; use text_column as the text field (e.g. 'content'); skip empty; optional category_column as JSON field for targets (missing → "unknown").
    """
    def __init__(
        self,
        dataset_path: Path,
        text_column: str,
        category_column: Optional[str] = None,
    ):
        path = Path(dataset_path)
        if path.is_file():
            self._load_csv(path, text_column, category_column)
        elif path.is_dir():
            self._load_json_dir(path, text_column, category_column)
        else:
            raise FileNotFoundError(f"Dataset path is neither a file nor a directory: {path}")

    def _load_csv(self, path: Path, text_column: str, category_column: Optional[str]) -> None:
        self.df = pd.read_csv(path)
        self.data = self.df[text_column].tolist()
        if category_column and category_column in self.df.columns:
            idx_mapping = {x: i for i, x in enumerate(self.df[category_column].unique())}
            self.target = [idx_mapping[x] for x in self.df[category_column]]
            self.target_names = {i: x for i, x in enumerate(self.df[category_column].unique())}
        else:
            self.target = [0] * len(self.data)
            self.target_names = {0: "unknown"}
        self.source_ids = (
            self.df["id"].astype(str).tolist()
            if "id" in self.df.columns
            else [str(i) for i in range(len(self.data))]
        )

    def _load_json_dir(
        self, path: Path, text_field: str, category_column: Optional[str] = None
    ) -> None:
        self.data = []
        self.source_ids = []
        raw_labels: list[str] = []
        for p in sorted(path.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    obj = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            text = (obj.get(text_field) or "").strip()
            if not text:
                continue
            self.data.append(text)
            self.source_ids.append(str(obj.get("_id", p.stem)))
            if category_column:
                val = obj.get(category_column)
                raw_labels.append(str(val).strip() if val is not None and str(val).strip() else "unknown")
            else:
                raw_labels.append("unknown")
        if raw_labels:
            unique = list(dict.fromkeys(raw_labels))  # preserve order
            idx_mapping = {x: i for i, x in enumerate(unique)}
            self.target = [idx_mapping[x] for x in raw_labels]
            self.target_names = {i: x for i, x in enumerate(unique)}
        else:
            self.target = []
            self.target_names = {0: "unknown"}
        self.df = pd.DataFrame({"doc_id": self.source_ids, text_field: self.data})


class NYTDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/ny_times_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['keyword'].unique())}
        self.target = [idx_mapping[x] for x in self.df['keyword']]
        self.target_names = {i: x for i, x in enumerate(self.df['keyword'].unique())}
        self.source_ids = [str(i) for i in range(len(self.data))]

class ArXivDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/arxiv_articles.csv')
        self.data = self.df['Summary'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['Category'].unique())}
        self.target = [idx_mapping[x] for x in self.df['Category']]
        self.target_names = {i: x for i, x in enumerate(self.df['Category'].unique())}
        self.source_ids = [str(i) for i in range(len(self.data))]
        
class PubmedDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/pubmed_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['mesh_subheading'].unique())}
        self.target = [idx_mapping[x] for x in self.df['mesh_subheading']]
        self.target_names = {i: x for i, x in enumerate(self.df['mesh_subheading'].unique())}
        self.source_ids = [str(i) for i in range(len(self.data))]


def get_nyt():
    return NYTDataset()

def get_arxiv():
    return ArXivDataset()

def get_pubmed():
    return PubmedDataset()


def get_dataset_from_csv(
    dataset_path: Path,
    text_column: str,
    category_column: Optional[str] = None,
):
    """Load dataset from CSV path and column names."""
    return GenericDataset(
        Path(dataset_path),
        text_column,
        category_column,
    )


if __name__ == '__main__':
    arxiv = get_arxiv()
    nytimes = get_nyt()
    print(nytimes.target_names)
