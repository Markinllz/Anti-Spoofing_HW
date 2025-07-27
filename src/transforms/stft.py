import torch
import torch.nn as nn
import torchaudio
from typing import Dict, Any


class STFTTransform(nn.Module):
    """
    Short-Time Fourier Transform (STFT) for audio processing.
    """

    def __init__(self, n_fft=1024, hop_length=512, win_length=1024, **kwargs):
        """
        Args:
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            win_length (int): Window size
            **kwargs: additional arguments
        """
        super(STFTTransform, self).__init__()
        
        print("🎵 Инициализация STFTTransform...")
        print(f"   📊 n_fft: {n_fft}")
        print(f"   📊 hop_length: {hop_length}")
        print(f"   📊 win_length: {win_length}")
        
        # Логируем дополнительные параметры
        for key, value in kwargs.items():
            print(f"   📊 {key}: {value}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        print("✅ STFTTransform инициализирован")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to audio signal.
        
        Args:
            audio (torch.Tensor): input audio tensor
            
        Returns:
            torch.Tensor: STFT spectrogram
        """
        # Логируем входные данные (только для отладки)
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   🎵 STFTTransform forward: audio shape={audio.shape}")
            print(f"      Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
            print(f"      Audio dtype: {audio.dtype}")
        
        # Убеждаемся, что аудио имеет правильную форму
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Добавляем batch dimension
        elif audio.dim() == 3:
            audio = audio.squeeze(1)  # Убираем лишний канал
        
        # Применяем STFT
        stft_output = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=torch.hann_window(self.win_length).to(audio.device)
        )
        
        # Конвертируем в спектрограмму
        spectrogram = torch.abs(stft_output)
        
        # Логируем выходные данные
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📊 STFT output shape: {stft_output.shape}")
            print(f"   📊 Spectrogram shape: {spectrogram.shape}")
            print(f"   📊 Spectrogram range: [{spectrogram.min().item():.4f}, {spectrogram.max().item():.4f}]")
        
        return spectrogram

    def set_debug_mode(self, debug_forward=False):
        """
        Включает режим отладки для логирования forward pass.
        
        Args:
            debug_forward (bool): логировать forward pass
        """
        self._debug_forward = debug_forward
        if debug_forward:
            print(f"🐛 Режим отладки включен для {self.__class__.__name__}")
            print(f"   🎵 Debug forward: {debug_forward}")


class MelSpectrogramTransform(nn.Module):
    """
    Mel Spectrogram transform for audio processing.
    """

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80, **kwargs):
        """
        Args:
            sample_rate (int): Audio sample rate
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of mel filter banks
            **kwargs: additional arguments
        """
        super(MelSpectrogramTransform, self).__init__()
        
        print("🎵 Инициализация MelSpectrogramTransform...")
        print(f"   📊 sample_rate: {sample_rate}")
        print(f"   📊 n_fft: {n_fft}")
        print(f"   📊 hop_length: {hop_length}")
        print(f"   📊 n_mels: {n_mels}")
        
        # Логируем дополнительные параметры
        for key, value in kwargs.items():
            print(f"   📊 {key}: {value}")
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Создаем mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window
        )
        
        print("✅ MelSpectrogramTransform инициализирован")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply Mel Spectrogram transform to audio signal.
        
        Args:
            audio (torch.Tensor): input audio tensor
            
        Returns:
            torch.Tensor: Mel spectrogram
        """
        # Логируем входные данные (только для отладки)
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   🎵 MelSpectrogramTransform forward: audio shape={audio.shape}")
            print(f"      Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
            print(f"      Audio dtype: {audio.dtype}")
        
        # Убеждаемся, что аудио имеет правильную форму
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Добавляем batch dimension
        elif audio.dim() == 3:
            audio = audio.squeeze(1)  # Убираем лишний канал
        
        # Применяем mel spectrogram transform
        mel_spectrogram = self.mel_transform(audio)
        
        # Логируем выходные данные
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📊 Mel spectrogram shape: {mel_spectrogram.shape}")
            print(f"   📊 Mel spectrogram range: [{mel_spectrogram.min().item():.4f}, {mel_spectrogram.max().item():.4f}]")
        
        return mel_spectrogram

    def set_debug_mode(self, debug_forward=False):
        """
        Включает режим отладки для логирования forward pass.
        
        Args:
            debug_forward (bool): логировать forward pass
        """
        self._debug_forward = debug_forward
        if debug_forward:
            print(f"🐛 Режим отладки включен для {self.__class__.__name__}")
            print(f"   🎵 Debug forward: {debug_forward}")