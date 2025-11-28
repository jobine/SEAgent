from abc import ABC, abstractmethod
from typing import List


class Benchmark(ABC):
    PASS = 'PASS'
    FAIL = 'FAIL'

    def __init__(self, name: str, data_path: str, log_path: str):
        self.name = name
        self.data_path = data_path
        self.log_path = log_path

        self._train_data: List[dict] | None = None
        self._validation_data: List[dict] | None = None
        self._test_data: List[dict] | None = None

        # Load the data
        self.load_data()

    @abstractmethod
    async def load_data(self):
        pass