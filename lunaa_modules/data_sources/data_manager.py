"""Data sources integration including HuggingFace datasets"""
try:
    from datasets import load_dataset
    import pandas as pd
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

class DataSourceManager:
    def __init__(self):
        self.loaded_datasets = {}
        
    def load_huggingface_dataset(self, dataset_name: str, split: str = 'train', limit: int = 100):
        """Load a dataset from HuggingFace"""
        if not _DATASETS_AVAILABLE:
            return "HuggingFace datasets library not installed"
        
        try:
            dataset = load_dataset(dataset_name, split=split)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            self.loaded_datasets[dataset_name] = dataset
            return f"Loaded {dataset_name} with {len(dataset)} examples"
        except Exception as e:
            return f"Error loading dataset: {e}"
    
    def query_dataset(self, dataset_name: str, query: str = None):
        """Query a loaded dataset"""
        if dataset_name not in self.loaded_datasets:
            return f"Dataset {dataset_name} not loaded"
        
        dataset = self.loaded_datasets[dataset_name]
        # Return first few examples
        examples = []
        for i, example in enumerate(dataset):
            if i >= 5:
                break
            examples.append(str(example))
        return "\n".join(examples)
    
    def get_free_api_data(self, api_name: str):
        """Placeholder for free API integrations"""
        # Could integrate: OpenWeatherMap, NewsAPI, etc.
        return f"Free API integration for {api_name} - to be implemented"
