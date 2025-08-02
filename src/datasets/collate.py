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

    # Обработка данных (может быть аудио или спектрограммы после STFT)
    data_tensors = [elem["data_object"] for elem in dataset_items]
    
    # Handle different tensor shapes
    if len(data_tensors) == 0:
        # Return empty batch
        result_batch["data_object"] = torch.empty(0)
        result_batch["labels"] = torch.empty(0, dtype=torch.long)
        return result_batch
    
    # Проверяем размерности - если 2D, то это спектрограммы, если 1D/2D - аудио
    first_tensor = data_tensors[0]
    
    if first_tensor.dim() >= 2 and first_tensor.shape[0] > 100:  # Спектрограмма
        # Для спектрограмм - padding по временной оси (последняя размерность)
        max_time = max(data.shape[-1] for data in data_tensors)
        
        padded_data = []
        for data in data_tensors:
            if data.shape[-1] < max_time:
                # Pad по времени
                padding = max_time - data.shape[-1]
                data = F.pad(data, (0, padding))
            padded_data.append(data)
    else:
        # Для аудио - padding как раньше
        max_length = max(data.shape[-1] for data in data_tensors)
        
        padded_data = []
        for data in data_tensors:
            if data.shape[-1] < max_length:
                padding = max_length - data.shape[-1]
                data = F.pad(data, (0, padding))
            padded_data.append(data)
    
    result_batch["data_object"] = torch.stack(padded_data)
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items], dtype=torch.long)

    return result_batch