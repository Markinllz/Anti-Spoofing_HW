import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from pathlib import Path


class AudioSpoofingDataset(BaseDataset):
    """
    Dataset class for ASVspoof2019 audio anti-spoofing challenge.
    """

    def __init__(self, name, label_path, audio_path, out_path, instance_transforms=None, max_samples=None, *args, **kwargs):
        """
        Args:
            name (str): partition name (train, dev, eval)
            label_path (str): path to the label file
            audio_path (str): path to the audio directory
            out_path (str): path to save the processed index
            instance_transforms (callable): transforms to apply to each instance
            max_samples (int): maximum number of samples for debugging (None for all)
        """
        self.name = name
        self.label_path = label_path
        self.audio_path = audio_path
        self.out_path = out_path
        self.max_samples = max_samples
        
        # Create index if it doesn't exist
        if Path(out_path).exists():
            print(f"Загружаем готовый index.json: {out_path}")
            index = read_json(out_path)
            print(f"Загружено {len(index)} записей из кэша")
            
            # Автоматически заменяем пути для Kaggle
            kaggle_data_path = os.environ.get("DATA_PATH")
            if kaggle_data_path and kaggle_data_path != "data":
                print(f"Заменяем пути для Kaggle: {kaggle_data_path}")
                for item in index:
                    # Заменяем старые пути на новые для Kaggle
                    if item["path"].startswith("data/ASVspoof2019_LA_"):
                        # Извлекаем имя файла из старого пути
                        file_name = item["path"].split("/")[-1]
                        # Определяем тип данных (train/dev/eval)
                        if "train" in item["path"]:
                            item["path"] = f"{kaggle_data_path}/ASVspoof2019_LA_train/flac/{file_name}"
                        elif "dev" in item["path"]:
                            item["path"] = f"{kaggle_data_path}/ASVspoof2019_LA_dev/flac/{file_name}"
                        elif "eval" in item["path"]:
                            item["path"] = f"{kaggle_data_path}/ASVspoof2019_LA_eval/flac/{file_name}"
        else:
            print(f"Создаем новый index.json: {out_path}")
            index = self._create_index(label_path, audio_path, out_path)

        # Ограничиваем размер датасета для отладки
        if self.max_samples is not None:
            index = index[:self.max_samples]

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
            # Return zero tensor as fallback - увеличиваем до 4 секунд
            fallback_waveform = torch.zeros(1, 64000)  # 4 seconds of silence at 16kHz
            print(f"Ошибка загрузки аудио {audio_path}: {e}")
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
        
        # Выводим статистику датасета
        total_samples = len(index)
        print(f"\nСтатистика датасета '{self.name}':")
        print(f"   Всего файлов: {total_samples}")
        print(f"   Bonafide (класс 0): {bonafide_count} ({100*bonafide_count/total_samples:.1f}%)")
        print(f"   Spoof (класс 1): {spoof_count} ({100*spoof_count/total_samples:.1f}%)")
        print(f"   Соотношение spoof/bonafide: {spoof_count/bonafide_count:.2f}")

        return index
