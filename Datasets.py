import pandas as pd
from pathlib import Path
from typing import Optional


class GenericDataset:
    """Generic dataset loaded from a CSV path and column names (no metadata file)."""
    def __init__(
        self,
        dataset_path: Path,
        text_column: str,
        category_column: Optional[str] = None,
    ):
        self.df = pd.read_csv(dataset_path)
        self.data = self.df[text_column].tolist()
        if category_column and category_column in self.df.columns:
            idx_mapping = {x: i for i, x in enumerate(self.df[category_column].unique())}
            self.target = [idx_mapping[x] for x in self.df[category_column]]
            self.target_names = {i: x for i, x in enumerate(self.df[category_column].unique())}
        else:
            self.target = [0] * len(self.data)
            self.target_names = {0: "unknown"}


class NYTDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/ny_times_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['keyword'].unique())}
        self.target = [idx_mapping[x] for x in self.df['keyword']]
        self.target_names = {i: x for i, x in enumerate(self.df['keyword'].unique())}

class ArXivDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/arxiv_articles.csv')
        self.data = self.df['Summary'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['Category'].unique())}
        self.target = [idx_mapping[x] for x in self.df['Category']]
        self.target_names = {i: x for i, x in enumerate(self.df['Category'].unique())}
        
class PubmedDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/pubmed_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['mesh_subheading'].unique())}
        self.target = [idx_mapping[x] for x in self.df['mesh_subheading']]
        self.target_names = {i: x for i, x in enumerate(self.df['mesh_subheading'].unique())}


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
