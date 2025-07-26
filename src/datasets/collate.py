import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # Pad audio sequences to the same length
    audio_tensors = [elem["data_object"] for elem in dataset_items]
    max_length = max(audio.shape[-1] for audio in audio_tensors)
    
    padded_audio = []
    for audio in audio_tensors:
        if audio.shape[-1] < max_length:
            # Pad with zeros
            padding = max_length - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        padded_audio.append(audio)
    
    result_batch["data_object"] = torch.stack(padded_audio)
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch