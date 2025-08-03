import torch
import torch.nn as nn
import torchaudio.transforms as T


class LFCCTransform(nn.Module):
    """
    Linear Frequency Cepstral Coefficients (LFCC) transform для anti-spoofing.
    
    Использует официальную реализацию torchaudio.transforms.LFCC согласно параметрам из статьи:
    - 60-dimensional LFCC (20 static + 20 delta + 20 delta-delta)
    - Frame length: 20ms, frame shift: 10ms
    - 512-point FFT
    - 20 linearly spaced triangular filter bank
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=160,
                 win_length=320,
                 n_filter=20,
                 n_lfcc=20,
                 use_deltas=True,
                 **kwargs):
        """
        Args:
            sample_rate (int): Audio sample rate (16000)
            n_fft (int): FFT size (512)
            hop_length (int): Frame shift in samples (160 = 10ms at 16kHz)
            win_length (int): Frame length in samples (320 = 20ms at 16kHz)
            n_filter (int): Number of linear filter banks (20)
            n_lfcc (int): Number of static LFCC coefficients (20)
            use_deltas (bool): Whether to add delta and delta-delta coefficients
        """
        super(LFCCTransform, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.use_deltas = use_deltas
        
        self.lfcc = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            f_min=0.0,
            f_max=sample_rate // 2,
            n_lfcc=n_lfcc,
            dct_type=2,
            norm='ortho',
            log_lf=False,
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "center": True,
                "pad_mode": "reflect",
                "power": 2.0
            }
        )
        
        if use_deltas:
            self.compute_deltas = T.ComputeDeltas(win_length=5, mode='replicate')
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Применение LFCC трансформации к аудио сигналу.
        
        Args:
            audio (torch.Tensor): входной аудио тензор [batch, samples] или [samples]
            
        Returns:
            torch.Tensor: LFCC признаки [batch, n_lfcc_total, time_frames]
                         где n_lfcc_total = 60 (20 static + 20 delta + 20 delta-delta)
                         или 20 (только static) если use_deltas=False
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        
        lfcc_static = self.lfcc(audio)
        
        if not self.use_deltas:
            return lfcc_static
        
        delta_lfcc = self.compute_deltas(lfcc_static)
        delta_delta_lfcc = self.compute_deltas(delta_lfcc)
        
        lfcc_features = torch.cat([lfcc_static, delta_lfcc, delta_delta_lfcc], dim=1)
        
        return lfcc_features 