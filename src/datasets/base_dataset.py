import logging
import random
from typing import List, Dict, Any, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        instance_transforms: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            index (List[Dict[str, Any]]): list of dictionaries, each containing
                the data for one sample.
            instance_transforms (Optional[Dict[str, Any]]): transforms to apply
                to instances. Depend on the tensor name.
        """
        self.index = index
        self.instance_transforms = instance_transforms

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): index of the item.

        Returns:
            dict: item data.
        """
        item = self.index[idx]
        
        # Применяем instance transforms
        if self.instance_transforms is not None:
            item = self._apply_instance_transforms(item)
        
        return item

    def _apply_instance_transforms(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply instance transforms to the item.

        Args:
            item (Dict[str, Any]): item data.

        Returns:
            Dict[str, Any]: transformed item data.
        """
        for transform_name, transform in self.instance_transforms.items():
            if transform_name in item:
                try:
                    item[transform_name] = transform(item[transform_name])
                except Exception as e:
                    print(f"Ошибка в трансформе {transform_name}: {e}")
                    continue
        
        return item

    def load_object(self, path):
        """
        Load audio object from disk.

        Args:
            path (str): path to the audio file.
        Returns:
            data_object (Tensor): audio tensor
        """
        try:
            waveform, sample_rate = torchaudio.load(path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform
        except Exception as e:
            logger.error(f"Error loading audio file {path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, 16000)  # 1 second of silence at 16kHz

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name in instance_data:
                    instance_data[transform_name] = self.instance_transforms[
                        transform_name
                    ](instance_data[transform_name])
        return instance_data
    
    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list): list of records to filter.

        Returns:
            list: filtered list of records.
        """
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Assert that the index is valid.

        Args:
            index (list): list of records to validate.
        """
        assert isinstance(index, list), "Index should be a list"
        assert len(index) > 0, "Index should not be empty"
        for record in index:
            assert isinstance(record, dict), "Each record should be a dict"
            assert "path" in record, "Each record should have a 'path' field"
            assert "label" in record, "Each record should have a 'label' field"

    @staticmethod
    def _sort_index(index):
        """
        Sort the index by some criterion.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list): list of records to sort.

        Returns:
            list: sorted list of records.
        """
        return index

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle and limit the index.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list): list of records to shuffle and limit.
            limit (int): maximum number of records to keep.
            shuffle_index (bool): whether to shuffle the index.

        Returns:
            list: shuffled and limited list of records.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index