import torch

def audio_frontend(waveform):
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    stft = torch.stft(
        waveform.squeeze(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True
    )
    log_magnitude = torch.log(torch.abs(stft) + 1e-8)
    return log_magnitude.unsqueeze(0)