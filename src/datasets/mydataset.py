import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from pathlib import Path


class AudioSpoofingDataset(BaseDataset):
    """
    Dataset class for ASVspoof2019 audio anti-spoofing challenge.
    """

    def __init__(
        self, name="train", label_path=None, audio_path=None, out_path=None, 
        instance_transforms=None, max_samples=None, *args, **kwargs
    ):
        """
        Args:
            name (str): partition name (train, val, test)
            label_path (str): path to the protocol file
            audio_path (str): path to the directory with .flac files
            out_path (str): where to save index.json
            instance_transforms (dict): transforms to apply to instances
            max_samples (int): maximum number of samples for debugging (None for all)
        """
        self.name = name
        self.label_path = label_path
        self.audio_path = audio_path
        self.out_path = out_path
        self.max_samples = max_samples
        
        # Create index if it doesn't exist
        if Path(out_path).exists():
            index = read_json(out_path)
        else:
            index = self._create_index(label_path, audio_path, out_path)

        # Ограничиваем размер датасета для отладки
        if max_samples is not None:
            index = index[:max_samples]

        super().__init__(index, instance_transforms=instance_transforms, *args, **kwargs)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): index of the item.

        Returns:
            dict: item data with 'data_object' and 'labels' keys.
        """
        item = self.index[idx]
        
        # Load audio file
        audio_path = item["path"]
        label = item["label"]
        
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Create item with correct keys
            item_data = {
                "data_object": waveform,
                "labels": label
            }
            
            # Apply instance transforms
            if self.instance_transforms is not None:
                item_data = self._apply_instance_transforms(item_data)
            
            return item_data
            
        except Exception as e:
            # Return zero tensor as fallback
            fallback_waveform = torch.zeros(1, 16000)  # 1 second of silence at 16kHz
            return {
                "data_object": fallback_waveform,
                "labels": label
            }

    def _create_index(self, label_path, audio_path, out_path):
        """
        Args:
        label_path (str or Path): path to the protocol file (e.g., train.trn.txt)
        audio_path (str or Path): path to the directory with .flac files
        out_path (str or Path): where to save index.json

        Returns:
            index (list[dict]): list of dictionaries, each with "path" and "label" fields
        """
        index = []
        
        # Подсчитываем общее количество строк в файле
        with open(label_path, "r") as f:
            total_lines = sum(1 for _ in f)
        
        bonafide_count = 0
        spoof_count = 0

        with open(label_path, "r") as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Обработка строк")):
                parts = line.strip().split()
                file_id = parts[1]
                class_name = parts[-1]
                label = 0 if class_name == "bonafide" else 1  # Fixed typo
                path = str(Path(audio_path) / f"{file_id}.flac")
                
                # Проверяем существование файла
                if not Path(path).exists():
                    continue
                
                index.append(
                    {
                        "path" : path,
                        "label" : label
                    }
                )
                
                # Подсчитываем статистику
                if label == 0:
                    bonafide_count += 1
                else:
                    spoof_count += 1
        
        write_json(index, out_path)

        return index
