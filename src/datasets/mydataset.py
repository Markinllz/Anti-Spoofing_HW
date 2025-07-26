import numpy as np
import torch
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
        instance_transforms=None, *args, **kwargs
    ):
        """
        Args:
            name (str): partition name (train, val, test)
            label_path (str): path to the protocol file
            audio_path (str): path to the directory with .flac files
            out_path (str): where to save index.json
            instance_transforms (dict): transforms to apply to instances
        """
        self.name = name
        self.label_path = label_path
        self.audio_path = audio_path
        self.out_path = out_path
        
        # Create index if it doesn't exist
        if Path(out_path).exists():
            index = read_json(out_path)
        else:
            index = self._create_index(label_path, audio_path, out_path)

        super().__init__(index, instance_transforms=instance_transforms, *args, **kwargs)

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

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                class_name = parts[-1]
                label = 0 if class_name == "bonafide" else 1  # Fixed typo
                path = str(Path(audio_path) / f"{file_id}.flac")
                index.append(
                    {
                        "path" : path,
                        "label" : label
                    }
                )
        print("Separate to path and labels complete")
        write_json(index, out_path)

        print(f"Created {len(index)} entries in {out_path}")

        return index
