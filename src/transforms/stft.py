import torch
import torch.nn as nn
import torchaudio.transforms as T


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
    STFT transform for audio processing.
    """

    def __init__(self, n_fft=512, hop_length=None, win_length=None, **kwargs):
        """
        Args:
            n_fft (int): размер FFT
            hop_length (int): шаг FFT
            win_length (int): размер окна
            **kwargs: дополнительные аргументы
        """
        super(STFTTransform, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to audio signal.
        
        Args:
            audio (torch.Tensor): input audio tensor
            
        Returns:
            torch.Tensor: STFT spectrogram
        """

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Применяем STFT
        stft_output = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=torch.hann_window(self.win_length).to(audio.device)
        )
        
        spectrogram = torch.abs(stft_output)
        
        return spectrogram


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