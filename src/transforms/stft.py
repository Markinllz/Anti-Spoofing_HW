import torch
import torch.nn as nn
import librosa
import numpy as np


class LogTransform(nn.Module):
    """
    Логарифмическая трансформация для спектрограмм.
    Критически важна для аудио задач.
    """
    
    def __init__(self, eps=1e-6):
        super(LogTransform, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): входная спектрограмма
        Returns:
            torch.Tensor: логарифмированная спектрограмма
        """
        return torch.log(x + self.eps)


class STFTTransform(nn.Module):
    """
    STFT (Short-Time Fourier Transform) transform для anti-spoofing.
    Использует librosa для извлечения спектрограммы согласно статье.
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=160,
                 win_length=320,
                 **kwargs):
        """
        Args:
            sample_rate (int): Audio sample rate (16000)
            n_fft (int): FFT size (512)
            hop_length (int): Frame shift in samples (160 = 10ms at 16kHz)
            win_length (int): Frame length in samples (320 = 20ms at 16kHz)
        """
        super(STFTTransform, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Применение STFT трансформации к аудио сигналу.
        
        Args:
            audio (torch.Tensor): входной аудио тензор [batch, samples] или [samples]
            
        Returns:
            torch.Tensor: STFT признаки [batch, n_freq_bins, time_frames]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Конвертируем в numpy для librosa
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        batch_size = audio_np.shape[0]
        stft_features = []
        
        for i in range(batch_size):
            # Извлекаем STFT с librosa
            stft = librosa.stft(
                audio_np[i], 
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=True,
                pad_mode='reflect'
            )
            
            # Берем magnitude spectrum (мощность) - stft это numpy array
            magnitude = np.abs(stft)
            
            # Логарифмируем (log power spectrum как в статье)
            # Добавляем ε чтобы избежать логарифма от нуля
            eps = 1e-8
            log_magnitude = np.log(magnitude + eps)
            
            stft_features.append(log_magnitude)
        
        # Конвертируем обратно в torch tensor
        stft_tensor = torch.from_numpy(np.stack(stft_features)).float()
        
        return stft_tensor


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
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply Mel Spectrogram transform to audio signal.
        
        Args:
            audio (torch.Tensor): input audio tensor
            
        Returns:
            torch.Tensor: Mel spectrogram
        """
       
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        
       
        mel_spectrogram = self.mel_spectrogram(audio)
        
        return mel_spectrogram