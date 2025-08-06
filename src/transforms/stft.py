import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
    
    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        raise ValueError("Invalid params")
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        raise ValueError("Invalid params")
    y = torch.zeros_like(x, dtype=torch.float32)
    if a < b:
        index = torch.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:    
        index = torch.logical_and(b < x, x < c)              
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y 


def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)
    
    output
    ------
    output (batch, Length, dim)
    
    Delta is calculated along Length dimension
    """
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = F.pad(x.unsqueeze(1), (0, 0, 1, 1), 
                   'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:,2:]
    return output


def linear_fb(fn, sr, filter_num):
    """linear_fb(fn, sr, filter_num)
    create linear filter bank based on trim

    input
    -----
      fn: int, FFT points
      sr: int, sampling rate (Hz)
      filter_num: int, number of filters in filter-bank
    
    output
    ------
      fb: tensor, (fn//2+1, filter_num)

    Note that this filter bank is supposed to be used on 
    spectrum of dimension fn//2+1.

    See example in LFCC.
    """
    # build the triangle filter bank
    f = (sr / 2) * torch.linspace(0, 1, fn//2+1)
    filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
    filter_bank = torch.zeros([fn//2+1, filter_num])
    for idx in range(filter_num):
        filter_bank[:, idx] = trimf(
            f, [filter_bands[idx], 
                filter_bands[idx+1], 
                filter_bands[idx+2]])
    return filter_bank


class LinearDCT(nn.Module):
    """Linear DCT layer"""
    def __init__(self, input_dim, type='dct', norm='ortho'):
        super(LinearDCT, self).__init__()
        self.input_dim = input_dim
        self.type = type
        self.norm = norm
        
        # Create DCT matrix
        self.register_buffer('dct_matrix', self._create_dct_matrix())
        
    def _create_dct_matrix(self):
        """Create DCT matrix"""
        N = self.input_dim
        dct_matrix = torch.zeros(N, N)
        
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = 1.0 / np.sqrt(N)
                else:
                    dct_matrix[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        
        return dct_matrix
    
    def forward(self, x):
        """Apply DCT transformation"""
        # x: (batch, time, features)
        batch_size, time_steps, features = x.shape
        
        # Apply DCT to each time step
        x_reshaped = x.view(-1, features)  # (batch*time, features)
        dct_output = torch.matmul(x_reshaped, self.dct_matrix.T)
        dct_output = dct_output.view(batch_size, time_steps, -1)
        
        return dct_output


class LFCC(nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_LFB=False,
                 num_coef=None, min_freq=0, max_freq=1):
        """ Initialize LFCC
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank

          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta
        
          for_LFB: bool (default False), reserved for LFB feature
          num_coef: int or None, number of coeffs to be taken from filter bank.
                    Note that this is only used for LFCC, i.e., for_LFB=False
                    When None, num_coef will be equal to filter_num
          min_freq: float (default 0), min_freq * sr // 2 will be the minimum 
                    frequency of extracted FFT spectrum
          max_freq: float (default 1), max_freq * sr // 2 will be the maximum 
                    frequency of extracted FFT spectrum
        """
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef

        # decide the range of frequency bins
        if min_freq >= 0 and min_freq < max_freq and max_freq <= 1:
            self.min_freq_bin = int(min_freq * (fn//2+1))
            self.max_freq_bin = int(max_freq * (fn//2+1))
            self.num_fft_bins = self.max_freq_bin - self.min_freq_bin 
        else:
            print("LFCC cannot work with min_freq {:f} and max_freq {:}".format(
                min_freq, max_freq))
            raise ValueError("Invalid frequency range")
        
        # build the triangle filter bank
        f = (sr / 2) * torch.linspace(min_freq, max_freq, self.num_fft_bins)
        filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
        filter_bank = torch.zeros([self.num_fft_bins, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx], 
                    filter_bands[idx+1], 
                    filter_bands[idx+2]])
        self.register_buffer('lfcc_fb', filter_bank)

        # DCT as a linear transformation layer
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')

        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_LFB = flag_for_LFB
        if self.num_coef is None:
            self.num_coef = filter_num

        # Add a buf to store window coefficients
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            # to avoid side effect
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        x_stft = torch.stft(x_copy, self.fn, self.fs, self.fl, 
                           window=self.window_buf, 
                           onesided=True, return_complex=True)

        # amplitude
        sp_amp = torch.abs(x_stft).pow(2).permute(0, 2, 1).contiguous()
        
        if self.min_freq_bin > 0 or self.max_freq_bin < (self.fn//2+1):
            sp_amp = sp_amp[:, :, self.min_freq_bin:self.max_freq_bin]
        
        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) + 
                                 torch.finfo(torch.float32).eps)
        
        # DCT (if necessary, remove DCT)
        lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature
        
        # Truncate the output of l_dct when necessary
        if not self.flag_for_LFB and self.num_coef != self.filter_num:
            lfcc = lfcc[:, :, :self.num_coef]
            

        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output


class LFB(LFCC):
    """ Linear filterbank feature
      Chen, T., Kumar, A., Nagarsheth, P., Sivaraman, G. & Khoury, E. 
      Generalization of Audio Deepfake Detection. in Proc. Odyssey 132-137 
      (2020). doi:10.21437/Odyssey.2020-19 
       
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=False):
        """ Initialize LFB
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank
          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta
        """
        super(LFB, self).__init__(fl, fs, fn, sr, filter_num, with_energy,
                                  with_emphasis, with_delta, flag_for_LFB=True)
        return
    
    def forward(self, x):
        """
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfb_output: tensor(batch, frame_num, dim_num)
        """
        return super(LFB, self).forward(x)


class Spectrogram(nn.Module):
    """ Spectrogram front-end
    """
    def __init__(self, fl, fs, fn, sr, 
                 with_emphasis=True, with_delta=False, in_db=False):
        """ Initialize Spectrogram
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default False), whether use delta and delta-delta
          in_db: bool, (default False), use 20log10(amp)? if False, use amp
        """
        super(Spectrogram, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr

        # opts
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.in_db = in_db

        # buf to store window coefficients
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         sp_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl, 
                           window=self.window_buf, 
                           onesided=True, return_complex=True)        

        # amplitude
        sp_amp = torch.abs(x_stft).pow(2).permute(0, 2, 1).contiguous()
        
        if self.in_db:
            sp_amp = torch.log10(sp_amp + torch.finfo(torch.float32).eps)

        # Add delta coefficients
        if self.with_delta:
            sp_delta = delta(sp_amp)
            sp_delta_delta = delta(sp_delta)
            sp_output = torch.cat((sp_amp, sp_delta, sp_delta_delta), 2)
        else:
            sp_output = sp_amp

        # done
        return sp_output


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) transformation.
    """
    
    def __init__(self, 
                 n_fft: int = 512,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 normalized: bool = False,
                 onesided: bool = True,
                 return_complex: bool = True):
        """
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between STFT windows
            win_length: Size of STFT window
            window: Window function type
            center: Whether to pad the signal
            pad_mode: Padding mode
            normalized: Whether to normalize the STFT
            onesided: Whether to return one-sided spectrum
            return_complex: Whether to return complex values
        """
        super(STFT, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.return_complex = return_complex
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, time_steps)
        Returns:
            stft: STFT tensor of shape (batch_size, freq_bins, time_frames, 2)
        """
        # Apply STFT
        stft = torch.stft(
            x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(x.device),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=self.return_complex
        )
        
        return stft