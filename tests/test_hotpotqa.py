"""Unit tests for HotpotQA benchmark."""

import os
import tempfile
import pytest
from unittest.mock import patch

from benchmarks.hotpotqa import HotpotQA
from benchmarks.benchmark import DatasetType


class TestHotpotQA:
    """Test cases for HotpotQA class."""

    @pytest.fixture
    def temp_data_folder(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_benchmarks_json(self):
        """Mock benchmarks.json content."""
        return {
            "hotpotqa": {
                "train": {
                    "url": "http://example.com/train.json",
                    "name": "hotpotqa_train.json"
                },
                "validate": {
                    "url": "http://example.com/validate.json",
                    "name": "hotpotqa_validate.json"
                }
            }
        }

    @pytest.fixture
    def mock_train_data(self):
        """Mock training data."""
        return [
            {
                "_id": "test_id_1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "type": "bridge",
                "level": "easy"
            },
            {
                "_id": "test_id_2",
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
                "type": "comparison",
                "level": "medium"
            }
        ]

    @pytest.fixture
    def mock_validate_data(self):
        """Mock validation data."""
        return [
            {
                "_id": "val_id_1",
                "question": "What is 2 + 2?",
                "answer": "4",
                "type": "bridge",
                "level": "easy"
            }
        ]

    def test_init_default_data_folder(self, mock_benchmarks_json, mock_train_data, mock_validate_data):
        """Test HotpotQA initialization with default data folder."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA()
            
            assert hotpotqa.name == 'hotpotqa'
            assert hotpotqa.data_folder == os.path.expanduser('~/.seagent/benchmarks')

    def test_init_custom_data_folder(self, temp_data_folder, mock_benchmarks_json, mock_train_data, mock_validate_data):
        """Test HotpotQA initialization with custom data folder."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder)
            
            assert hotpotqa.name == 'hotpotqa'
            assert hotpotqa.data_folder == temp_data_folder

    def test_init_with_train_dataset_type(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test HotpotQA initialization with TRAIN dataset type only."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.TRAIN)
            
            assert hotpotqa.dataset_type == DatasetType.TRAIN
            assert hotpotqa._train_data == mock_train_data
            assert hotpotqa._validate_data is None

    def test_init_with_validate_dataset_type(self, temp_data_folder, mock_benchmarks_json, mock_validate_data):
        """Test HotpotQA initialization with VALIDATE dataset type only."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.VALIDATE)
            
            assert hotpotqa.dataset_type == DatasetType.VALIDATE
            assert hotpotqa._validate_data == mock_validate_data
            assert hotpotqa._train_data is None

    def test_load_data_downloads_when_file_not_exists(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test that load_data downloads files when they don't exist."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=False):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                return mock_train_data
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.TRAIN)
            
            # Verify download_file was called
            assert mock_download.called

    def test_load_data_skips_download_when_file_exists(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test that load_data skips download when files already exist."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                return mock_train_data
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.TRAIN)
            
            # Verify download_file was not called
            assert not mock_download.called

    def test_load_data_force_reload(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test that load_data with force_reload=True re-downloads files."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            call_count = [0]
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                return mock_train_data
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.TRAIN)
            
            # Reset mock to track calls during force reload
            mock_download.reset_mock()
            
            # Force reload
            hotpotqa.load_data(force_reload=True)
            
            # Verify download_file was called during force reload
            assert mock_download.called

    def test_load_data_benchmark_not_found(self, temp_data_folder):
        """Test that load_data raises ValueError when benchmark is not in benchmarks.json."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download:
            
            # Return empty benchmarks.json (no hotpotqa entry)
            mock_load_json.return_value = {}
            
            with pytest.raises(ValueError, match="Benchmark hotpotqa not found"):
                HotpotQA(data_folder=temp_data_folder)

    def test_train_data_property(self, temp_data_folder, mock_benchmarks_json, mock_train_data, mock_validate_data):
        """Test train_data property returns correct data."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder)
            
            assert hotpotqa.train_data == mock_train_data
            assert len(hotpotqa.train_data) == 2

    def test_validate_data_property(self, temp_data_folder, mock_benchmarks_json, mock_train_data, mock_validate_data):
        """Test validate_data property returns correct data."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder)
            
            assert hotpotqa.validate_data == mock_validate_data
            assert len(hotpotqa.validate_data) == 1

    def test_train_data_property_raises_when_not_loaded(self, temp_data_folder, mock_benchmarks_json, mock_validate_data):
        """Test train_data property raises ValueError when not loaded."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'validate' in path:
                    return mock_validate_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            # Only load VALIDATE dataset
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.VALIDATE)
            
            with pytest.raises(ValueError, match="Train data not loaded"):
                _ = hotpotqa.train_data

    def test_test_data_property_raises_when_not_loaded(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test test_data property raises ValueError when not loaded (hotpotqa has no test set)."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.TRAIN)
            
            with pytest.raises(ValueError, match="Test data not loaded"):
                _ = hotpotqa.test_data

    def test_load_data_with_none_dataset(self, temp_data_folder, mock_benchmarks_json, mock_train_data):
        """Test _load_data returns None when dataset is None."""
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                elif 'train' in path:
                    return mock_train_data
                return []
            
            mock_load_json.side_effect = load_json_side_effect
            
            hotpotqa = HotpotQA(data_folder=temp_data_folder, dataset_type=DatasetType.ALL)
            
            # HotpotQA doesn't have test data in benchmarks.json
            assert hotpotqa._test_data is None


class TestHotpotQAEvaluate:
    """Test cases for HotpotQA evaluate method."""

    @pytest.fixture
    def hotpotqa_instance(self, tmp_path):
        """Create a HotpotQA instance for testing evaluate method."""
        mock_benchmarks_json = {
            "hotpotqa": {
                "train": {
                    "url": "http://example.com/train.json",
                    "name": "hotpotqa_train.json"
                }
            }
        }
        mock_train_data = [{"question": "test", "answer": "test"}]
        
        with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
             patch('benchmarks.hotpotqa.download_file') as mock_download, \
             patch('os.path.exists', return_value=True):
            
            def load_json_side_effect(path):
                if 'benchmarks.json' in path:
                    return mock_benchmarks_json
                return mock_train_data
            
            mock_load_json.side_effect = load_json_side_effect
            
            return HotpotQA(data_folder=str(tmp_path), dataset_type=DatasetType.TRAIN)

    @pytest.mark.asyncio
    async def test_evaluate_returns_none(self, hotpotqa_instance):
        """Test evaluate method (currently returns None as placeholder)."""
        result = await hotpotqa_instance.evaluate("Paris", "Paris")
        # Currently evaluate is not implemented, returns None
        assert result is None


class TestHotpotQAIntegration:
    """Integration tests for HotpotQA (requires actual file operations)."""

    def test_data_folder_created(self):
        """Test that data folder is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_folder = os.path.join(tmpdir, 'new_folder', 'benchmarks')
            
            with patch('benchmarks.hotpotqa.load_json') as mock_load_json, \
                 patch('benchmarks.hotpotqa.download_file'):
                
                # Mock load_json to return train data on second call
                mock_load_json.side_effect = [
                    {
                        "hotpotqa": {
                            "train": {
                                "url": "http://example.com/train.json",
                                "name": "hotpotqa_train.json"
                            }
                        }
                    },
                    [{"question": "test", "answer": "test"}]  # mock train data
                ]
                
                HotpotQA(data_folder=test_folder, dataset_type=DatasetType.TRAIN)
                
                assert os.path.exists(test_folder)
