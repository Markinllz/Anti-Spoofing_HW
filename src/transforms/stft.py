import torch
import torch.nn as nn


def audio_frontend(waveform):

    n_fft = 1024
    hop_length = 256
    win_length = 1024

    waveform = waveform.squeeze()

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=waveform.device),
        return_complex=True
    )

    magnitude = torch.abs(stft)
    log_magnitude = torch.log(magnitude + 1e-8)
    log_magnitude = log_magnitude.unsqueeze(0)
    return log_magnitude


class AudioFrontend(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, waveform):
      


        return audio_frontend(waveform)