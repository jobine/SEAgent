import os
from typing import Any
from .benchmark import Benchmark, DatasetType
from .tools import load_json, download_file


class HotpotQA(Benchmark):
    def __init__(self, data_folder: str = None, dataset_type: DatasetType = DatasetType.ALL):
        self.dataset_type = dataset_type

        location = os.path.normpath(os.path.expanduser(data_folder or '~/.seagent/benchmarks'))
        super().__init__(name=type(self).__name__.lower(), data_folder=location)

    def load_data(self, force_reload: bool = False) -> None:
        '''
        Load the HotpotQA dataset into the benchmark.
        Downloads the dataset files if they do not exist or if force_reload is True.
        '''

        name = type(self).__name__.lower()
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        benchmarks = load_json(os.path.join(module_dir, 'benchmarks.json'))

        if name in benchmarks:
            # Implement data loading logic here
            # For example, download dataset files if not present or force_reload is True
            # Then load the data into self._train_data, self._validate_data, self._test_data
            datasets = benchmarks[name]

            # Load training data
            if self.dataset_type in (DatasetType.ALL, DatasetType.TRAIN):
                self._train_data = self._load_data(dataset=datasets.get(DatasetType.TRAIN.value), force_reload=force_reload)
            
            # Load validation data
            if self.dataset_type in (DatasetType.ALL, DatasetType.VALIDATE):
                self._validate_data = self._load_data(dataset=datasets.get(DatasetType.VALIDATE.value), force_reload=force_reload)
            
            # Load test data
            if self.dataset_type in (DatasetType.ALL, DatasetType.TEST):
                self._test_data = self._load_data(dataset=datasets.get(DatasetType.TEST.value), force_reload=force_reload)
        else:
            raise ValueError(f'Benchmark {name} not found in benchmarks.json')
        
    def _load_data(self, dataset: dict | None, force_reload: bool = False) -> list[dict] | None:
        '''
        Load a specific dataset (train/validate/test) based on the provided dataset info.
        If the dataset file does not exist or force_reload is True, it downloads the dataset.
        '''

        if dataset is None:
            return None
        
        file_path = os.path.join(self.data_folder, dataset['name'])
        if not os.path.exists(file_path) or force_reload:
            download_file(url=dataset['url'], destination_path=file_path)
        
        data = load_json(file_path)
        return data
        
    async def evaluate(self, prediction: Any, label: Any) -> dict:
        pass
