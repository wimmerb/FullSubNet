import torch
import torch.nn as nn
from .helpers import replace_denormals


class ErrorToSignalRatioLoss(nn.Module):
    def __init__(self,
                 pre_emphasis=False,
                 pre_emphasis_coeff=0.95,
                 eps = 10e-10):
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.eps = 10e-10

    def pre_emphasis(self, x, coeff):
        return torch.cat(
                (x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), 
                dim=2
                )
        
    def forward(self, y_pred: torch.tensor, y: torch.tensor):
        if self.pre_emphasis:
            y = self.pre_emphasis(y, self.pre_emphasis_coeff)
            y_pred = self.pre_emphasis(y_pred, pre_emphasis_coeff)
        return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + self.eps)


class NegativeSNRLoss(nn.Module):
    """ 
    Negative Signal-to-Noise Ratio loss.

    Calculates the negative SNR over a predicted output and ground truth
    output pair.
    """
    def __init__(self, 
                 eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.eps = eps
    
    def forward(self, y_pred: torch.tensor, y: torch.tensor):
        """ Calculates the negative SNR loss based on a predicted output and a
        ground truth output. 

        Args:
            y_pred (torch.tensor): Predicted tensor containing the denoised 
                signal.

            y (torch.tensor): Ground truth tensor containing the clean signal.

        Returns:
            loss (torch.tensor): 1D tensor containing the loss function value
        """
        numerator = torch.sum(torch.square(y), dim=-1, keepdim=True)
        denominator = torch.sum(torch.square(y - y_pred), dim=-1, keepdim=True)
        loss = -10 * torch.log10(numerator / denominator + self.eps)
        # experimental based on 7 significant digits for torch.float32
        loss[torch.isneginf(loss)] = -140.0
        return torch.mean(loss, dim=0)


# TODO: Experimental
class GainMaskBasedNegativeSNRLoss(nn.Module):
    """ Negative Signal-to-Noise Ratio loss for gain mask based networks.

    Calculates the negative SNR over a predicted spectral mask and a complex
    stft output of a noisy speech signal and ground truth clean signal.
    """
    def __init__(self,
                 window_size: int = 512,
                 hop_size: int = 128,
                 eps: float = 1e-10):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.eps = eps
        self._window = torch.hann_window(self.window_size)
        self._negative_snr_loss = NegativeSNRLoss(eps=self.eps)

    # TODO: See what would be the best idea to keep the stft istft settings
    # consistants between the trained network and the loss and make them
    # network agnostic at the same time
    def istft(self, x_complex: torch.tensor):
        window = self._window.to(x_complex.device)
        
        istft = torch.istft(x_complex,
                            onesided=True,
                            center=True,
                            n_fft=self.window_size,
                            hop_length=self.hop_size,
                            normalized=False,
                            window=window)

        return istft

    def forward(self, y_pred_mask: torch.tensor, x_complex: torch.tensor, 
                y_complex: torch.tensor):
        """
        Calculates the negative SNR over a predicted spectral mask and a complex
        stft output of a noisy speech signal and ground truth clean signal.

        Args:
            y_pred_mask (torch.tensor): Predicted tensor containing the gain mask
                to be applied to the complex stft input x_complex.

            x_complex (torch.tensor): Tensor containing the complex stft of
                the input signal.

            y_complex (torch.tensor): Tensor containing the ground truth complex
                stft of the output signal.

        Returns:
            loss (torch.tensor): 1D tensor containing the loss function value
        """
        y_pred_complex = y_pred_mask.squeeze(1).permute(0, 2, 1) * x_complex
        y_pred = self.istft(y_pred_complex)
        y = self.istft(y_complex)

        return self._negative_snr_loss(y_pred, y)


# TODO: Fix the following:  each training sequence, 
# i. e. predicted and target signals, are
# normalized by the active target utterance level, to ensure balanced
# optimization for signal-level dependent losses
# NOTE: This didn't make much difference during overfitting a batch 
# but perhaps it is something it is worth trying
class ComplexCompressedMSELoss:
    def __init__(self,
                 c_: float = 0.3,
                 lambda_: float = 0.3,
                 eps: float = 1e-10):
        super().__init__()
        self.c_ = c_
        self.lambda_ = lambda_
        self.eps = eps

    # TODO: To be changed to helpers.replace_denormals
    def clean_denormals(self, input_tensor, threshold=1e-7):
        output_tensor = input_tensor.clone()
        output_tensor[(input_tensor < threshold) &
                      (input_tensor > -1.0 * threshold)] = threshold
        return output_tensor
 
    # TODO: To be changed to forward. Its parent class should be changed
    # for nn.Module before this
    def __call__(self, y_pred_mask, x_complex, y_complex):
        # clean denormals
        y_complex = self.clean_denormals(torch.real(y_complex)) + \
                                         1j * torch.imag(y_complex)

        # get target magnitude and phase
        y_mag = torch.abs(y_complex)
        y_phase = torch.angle(y_complex)

        # predicted complex stft
        y_pred_mask = y_pred_mask.squeeze(1).permute(0, 2, 1)
        y_pred_complex = y_pred_mask.type(torch.complex64) * x_complex

        # clean denormals
        y_pred_complex = self.clean_denormals(torch.real(y_pred_complex)) + \
                                              1j * torch.imag(y_pred_complex)

        # get predicted magnitude annd phase
        y_pred_mag = torch.abs(y_pred_complex)
        y_pred_phase = torch.angle(y_pred_complex)

        # target complex exponential        
        y_complex_exp = (y_mag ** self.c_).type(torch.complex64) * \
                torch.exp(1j * y_phase.type(torch.complex64))

        # predicted complex exponential
        y_pred_complex_exp = (y_pred_mag ** self.c_).type(torch.complex64) * \
                torch.exp(1j * y_pred_phase.type(torch.complex64))

        # magnitude only loss component
        mag_loss = torch.abs(y_mag ** self.c_ - y_pred_mag ** self.c_) ** 2
        mag_loss = torch.sum(mag_loss, dim=[1, 2])

        # complex loss component
        complex_loss = torch.abs(y_complex_exp - y_pred_complex_exp) ** 2
        complex_loss = torch.sum(complex_loss, dim=[1, 2])

        # blend both loss components
        loss = (1 - self.lambda_) * mag_loss + (self.lambda_) * complex_loss

        # returns the mean blended loss of the batch
        return torch.mean(loss)
