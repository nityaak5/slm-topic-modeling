import json
import pandas as pd
from pathlib import Path


class GenericDataset:
    """Generic dataset class that loads from metadata.json."""
    def __init__(self, metadata_path: Path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        dataset_path = Path(metadata['dataset_path'])
        self.df = pd.read_csv(dataset_path)
        self.data = self.df[metadata['text_column']].tolist()
        
        # Handle category column if present
        if 'category_column' in metadata and metadata['category_column']:
            category_col = metadata['category_column']
            idx_mapping = {x: i for i, x in enumerate(self.df[category_col].unique())}
            self.target = [idx_mapping[x] for x in self.df[category_col]]
            self.target_names = {i: x for i, x in enumerate(self.df[category_col].unique())}
        else:
            # No category column - create dummy targets (all 0)
            self.target = [0] * len(self.data)
            self.target_names = {0: 'unknown'}


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


def get_dataset_from_metadata(metadata_path: Path):
    """Load dataset from metadata.json file."""
    return GenericDataset(metadata_path)


if __name__ == '__main__':
    arxiv = get_arxiv()
    nytimes = get_nyt()
    print(nytimes.target_names)
