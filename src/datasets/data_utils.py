from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if batch_transforms is None:
        return
        
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, debug_mode=False):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        debug_mode (bool): if True, create minimal dataloaders for debugging.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # dataloaders init
    dataloaders = {}
    debug_subset = None  # Сохраняем один subset для всех разделов
    
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]
        
        # Для отладки все разделы используют одни и те же данные из train
        if debug_mode:
            from torch.utils.data import Subset
            if debug_subset is None:
                # Создаем subset только один раз из первого датасета (обычно train)
                debug_subset_indices = range(min(4, len(dataset)))
                debug_subset = Subset(dataset, debug_subset_indices)
                print(f"Debug mode: using {len(debug_subset)} samples for all partitions")
            dataset = debug_subset

        # Для отладки изменяем параметры даталоадера
        if debug_mode:
            batch_size = min(2, len(dataset))
            num_workers = 0
            pin_memory = False
            shuffle = False  # В debug режиме не перемешиваем для воспроизводимости
        else:
            batch_size = config.dataloader.batch_size
            num_workers = getattr(config.dataloader, 'num_workers', 4)
            pin_memory = getattr(config.dataloader, 'pin_memory', True)
            shuffle = True  # В обычном режиме обязательно перемешиваем

        assert batch_size <= len(dataset), (
            f"The batch size ({batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False,  # В debug режиме не отбрасываем данные
            shuffle=shuffle,
            worker_init_fn=set_worker_seed,
        )
        
        dataloaders[dataset_partition] = partition_dataloader
        
        if debug_mode:
            print(f"📁 {dataset_partition}: {len(dataset)} образцов, batch_size={batch_size}")

    return dataloaders, batch_transforms