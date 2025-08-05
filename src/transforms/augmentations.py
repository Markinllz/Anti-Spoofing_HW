import torch
import torch.nn.functional as F
import random
from typing import Tuple


class AddNoise:
    """Добавление шума - самая важная аугментация для anti-spoofing"""
    
    def __init__(self, noise_level_range: Tuple[float, float] = (0.001, 0.01), p: float = 0.5):
        """
        Args:
            noise_level_range (tuple): диапазон уровня шума
            p (float): вероятность применения
        """
        self.noise_level_range = noise_level_range
        self.p = p
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise_level = random.uniform(*self.noise_level_range)
            
            # Генерируем белый шум
            noise = torch.randn_like(waveform) * noise_level
            
            # Нормализуем амплитуду
            signal_power = torch.mean(waveform ** 2)
            noise_power = torch.mean(noise ** 2)
            
            if noise_power > 0:
                noise = noise * torch.sqrt(signal_power / noise_power) * noise_level
            
            return waveform + noise
        return waveform


class TimeStretch:
    """Изменение скорости воспроизведения - имитирует разные скорости речи"""
    
    def __init__(self, stretch_factor_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        """
        Args:
            stretch_factor_range (tuple): диапазон коэффициентов растяжения
            p (float): вероятность применения
        """
        self.stretch_factor_range = stretch_factor_range
        self.p = p
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            stretch_factor = random.uniform(*self.stretch_factor_range)
            
            # Изменяем длину через resample
            original_length = waveform.shape[-1]
            new_length = int(original_length / stretch_factor)
            
            resampled = F.interpolate(
                waveform.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            
            # Обрезаем или дополняем до исходной длины
            if resampled.shape[-1] > original_length:
                return resampled[:, :original_length]
            else:
                padding = original_length - resampled.shape[-1]
                return F.pad(resampled, (0, padding))
        
        return waveform


class TimeMasking:
    """Маскирование временных сегментов - имитирует потери в канале передачи"""
    
    def __init__(self, mask_ratio_range: Tuple[float, float] = (0.1, 0.3), p: float = 0.5):
        """
        Args:
            mask_ratio_range (tuple): диапазон доли маскируемого времени
            p (float): вероятность применения
        """
        self.mask_ratio_range = mask_ratio_range
        self.p = p
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            mask_ratio = random.uniform(*self.mask_ratio_range)
            mask_size = int(waveform.shape[-1] * mask_ratio)
            
            if mask_size > 0:
                max_start = waveform.shape[-1] - mask_size
                start_pos = random.randint(0, max_start)
                
                # Применяем маску (заменяем на нули)
                waveform = waveform.clone()
                waveform[:, start_pos:start_pos + mask_size] = 0
        
        return waveform


class ComposeAugmentations:
    """Композиция аугментаций"""
    
    def __init__(self, augmentations: list):
        self.augmentations = augmentations
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for augmentation in self.augmentations:
            waveform = augmentation(waveform)
        return waveform


# Предустановленные комбинации
def get_anti_spoofing_augmentations(p: float = 0.2) -> ComposeAugmentations:
    """2 основные аугментации для anti-spoofing: шум и маскинг"""
    return ComposeAugmentations([
        AddNoise(noise_level_range=(0.001, 0.01), p=p),
        TimeMasking(mask_ratio_range=(0.1, 0.3), p=p),
    ]) 