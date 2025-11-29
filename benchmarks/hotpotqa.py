from __future__ import annotations

import os
from .benchmark import Benchmark, DatasetType
from .tools import loads_json, download_file
from benchmarks import benchmark


class HotpotQA(Benchmark):
    def __init__(self, data_folder: str = None, dataset_type: DatasetType = DatasetType.ALL):
        self.dataset_type = dataset_type

        location = os.path.expanduser(data_folder or '~/.seagent/benchmarks')
        super().__init__(name=type(self).__name__.lower(), data_folder=location)

    def load_data(self, force_reload: bool = False) -> None:
        name = type(self).__name__.lower()
        benchmarks = loads_json('benchmarks/benchmarks.json')

        if name in benchmarks:
            # Implement data loading logic here
            # For example, download dataset files if not present or force_reload is True
            # Then load the data into self._train_data, self._validate_data, self._test_data
            datasets = benchmarks[name]

            if self.dataset_type in (DatasetType.ALL, DatasetType.TRAIN):
                # Load training data
                self._train_data = self._load_data(dataset=datasets.get(DatasetType.TRAIN.value), force_reload=force_reload)

            if self.dataset_type in (DatasetType.ALL, DatasetType.VALIDATE):
                # Load validation data
                self._validate_data = self._load_data(dataset=datasets.get(DatasetType.VALIDATE.value), force_reload=force_reload)
            if self.dataset_type in (DatasetType.ALL, DatasetType.TEST):
                # Load test data
                self._test_data = self._load_data(dataset=datasets.get(DatasetType.TEST.value), force_reload=force_reload)
        else:
            raise ValueError(f'Benchmark {name} not found in benchmarks.json')
        
    def _load_data(self, dataset: dict | None, force_reload: bool = False) -> list[dict] | None:
        if dataset is None:
            return None
        
        file_path = os.path.join(self.data_folder, dataset['name'])
        if not os.path.exists(file_path) or force_reload:
            download_file(url=dataset['url'], destination_path=file_path)
        
        data = loads_json(file_path)
        return data
        
    async def evaluate(self, prediction: str, label: str) -> str:
        pass


if __name__ == '__main__':
    hotpotqa = HotpotQA()
