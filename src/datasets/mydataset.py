import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from pathlib import Path


class AudioSpoofingDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.
    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            n_classes (int): number of classes.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "example" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

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
                label = 0 if class_name == "bonefide" else 1
                path = str(Path(audio_path) / f"{file_id}.flac")
                index.append(
                    {
                        "path" : path,
                        "label" : label
                    }
                )
        print("Seperate to path and labels complete")
        write_json(index, out_path)

        print(f"Created {len(index)} entries in {out_path}")

        return index
