import torch
import torch.nn as nn
from dtln_utils.custom_layers import CausalConv1d


class DTLN(nn.Module):
    """ Dual-Signal Transformation Long Short-Term Memory Network.

    Similar architecture as presented by N. Westhausen et. al for the
    DNS INTERSPEECH 2020:

    https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf

    The original implementation was done in tensorflow 2.x. The current
    implementation is based on it with some changes to attempt to improve
    the existing model for noise suppression tasks on speech signals.

    Args:
        sample_rate (int): Sample rate of the input files.
        window_size (int): Window size used to perform the STFT over 
            the input files.
        hop_size (int): Hop size used to perform the STFT over the input files.
        sample_duration (int): Duration in seconds of each input file.
        encoder_size (int): Channel output size of Conv1D layers used in the
            network (see architecture for the details).
        dropout_rate (float): Dropout rate used between LSTM) stacks.
        eps (float): Machine epsilon.
        learning_rate (float): Learning rate used to train the network.
        batch_size (int): Batch size.
        name (str): Network's name used to save checkpoints.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 window_size: int = 512,
                 hop_size: int = 128,
                 sample_duration: int = 15,
                 hidden_size: int = 128,
                 encoder_size: int = 256,
                 dropout_rate: float = 0.25,
                 eps: float = 1e-7,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 ):
        super().__init__()

        # audio parameters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_duration = sample_duration

        # amount of samples of each input audio data tensor
        self._chunk_samples = int(self.sample_duration * self.sample_rate)

        # machine epsilon
        self.eps = eps

        # network parameters
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # network first separation core
        self.layer_norm_11 = nn.LayerNorm(window_size // 2 + 1, eps=self.eps)
        self.lstm_11 = nn.LSTM(self.window_size // 2 + 1, self.hidden_size)
        self.dropout_11 = nn.Dropout(self.dropout_rate)
        self.lstm_12 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.fc_11 = nn.Linear(self.hidden_size, self.window_size // 2 + 1)
        self.afn_11 = nn.Sigmoid()

        # network second separation core
        self.conv1d_21 = nn.Conv1d(self.window_size,
                                   self.encoder_size,
                                   kernel_size=1,
                                   stride=1,
                                   bias=False)
        self.layer_norm_21 = nn.LayerNorm(self.encoder_size, eps=self.eps)
        self.lstm_21 = nn.LSTM(self.encoder_size, self.hidden_size)
        self.dropout_21 = nn.Dropout(self.dropout_rate)
        self.lstm_22 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.fc_21 = nn.Linear(self.hidden_size, self.encoder_size)
        self.afn_21 = nn.Sigmoid()

        # network additional final block
        self.causal_conv1d_22 = CausalConv1d(self.encoder_size,
                                             self.window_size,
                                             kernel_size=1,
                                             bias=False)
        self.overlap_and_add = nn.Fold((1, self._chunk_samples),
                                        kernel_size=(1, self.window_size),
                                        stride=(1, self.hop_size))

        self.init_weights_and_biases()

    def _init_lstm(self, lstm_layer):
        torch.nn.init.xavier_uniform_(lstm_layer.weight_ih_l0)
        torch.nn.init.xavier_uniform_(lstm_layer.weight_hh_l0)
        torch.nn.init.zeros_(lstm_layer.bias_ih_l0)
        torch.nn.init.zeros_(lstm_layer.bias_hh_l0)

    def _init_linear(self, linear_layer):
        torch.nn.init.xavier_uniform_(linear_layer.weight)
        torch.nn.init.zeros_(linear_layer.bias)

    def _init_layer_norm(self, layer_norm_layer):
        # TODO: Apparently this is the default init for these layers
        torch.nn.init.ones_(layer_norm_layer.weight)
        torch.nn.init.zeros_(layer_norm_layer.bias)

    # TODO: probably this decorator is not necessary considering the context
    # in which this function is run
    # NOTE: this function is a uniform way to access preprocessing + model
    # prediction from the evaluation script
    @torch.no_grad()
    def pred_fn(self, audio_tensor):
        pred_audio = self(audio_tensor)
        pred_audio = pred_audio.cpu().numpy().reshape(-1)
        return pred_audio

    def init_weights_and_biases(self):
        """ Manual initialization of weights. """
        # layer norm
        self._init_layer_norm(self.layer_norm_11)
        self._init_layer_norm(self.layer_norm_21)

        # fc init
        self._init_linear(self.fc_11)
        self._init_linear(self.fc_21)

        # lstm init
        self._init_lstm(self.lstm_11)
        self._init_lstm(self.lstm_12)
        self._init_lstm(self.lstm_21)
        self._init_lstm(self.lstm_22)

        # conv1d init
        torch.nn.init.xavier_uniform_(self.conv1d_21.weight)
        torch.nn.init.xavier_uniform_(self.causal_conv1d_22.weight)

    def stft(self, x: torch.tensor):
        """ Calculates the Short-Time Fourier Transform over a given tensor.

        The expected tensor is a 2D tensor in the following shape:
        (batch_size, audio_samples) e.g. a (4, 16000) would correspond to 
        a batch of 4 files of 1 second each at 16kHz of sample rate.

        Args:
            x (torch.tensor): Input tensor containing a single audio chunk
                or a batch of audio chunks. The last dimension is assumed to
                contain the raw audio samples.

        Returns:
            real, imag (torch.tensor, torch.tensor): A tuple with two torch
                tensors corresponding to the real and imaginary part of the
                STFT performed over the input tensor. Only the first half of
                the resulting bins are returned.
        """
        # hard coded hann window for now!
        hann_window = torch.hann_window(self.window_size).to(x.device)

        stft = torch.stft(x,
                          onesided=True,
                          center=False,
                          n_fft=self.window_size,
                          hop_length=self.hop_size,
                          normalized=False,
                          window=hann_window,
                          return_complex=True)

        return torch.abs(stft), torch.angle(stft)

    def ifft(self, x_mag: torch.tensor, x_phase: torch.tensor):
        """ Calculates the Inverse Fast Fourier Transform of the input 
        real and imaginary tensors.

        It assumes that the reconstruction is done using only the first half
        of the input features.

        Args:
            x_mag (torch.tensor): Magnitude input tensor.
            x_phase (torch.tensor): Phase input tensor.

                # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        """
        x_real = x_mag * torch.cos(x_phase)
        x_imag = x_mag * torch.sin(x_phase)
        x_complex = torch.complex(x_real, x_imag)
        ifft = torch.fft.irfft(x_complex, dim=-1)
        return ifft

    def forward(self, x, batch_size_special = None):
        """ DTLN forward pass. 

        Args:
            x (torch.tensor): Input tensor corresponding to a single audio file
                or a batch.
        """ 
        # (batch, samples) -> (batch, bins, frames)
        # e.g. (1, 240000) -> (1, 257, 1872), (1, 257, 1872) 
        x_mag, x_phase = self.stft(x) 
        print(x_mag.shape, x_phase.shape)

        # (batch, bins, frames) -> (batch, frames, bins)
        # e.g. (1, 257, 1872) -> (1, 1872, 257)
        # TODO: Probably unnecessary!
        # NOTE: This is used to multiply
        x_mag_ = x_mag.permute(0, 2, 1)

        # log norm (original uses Napierian log without scalar multiplier 10)
        x_mag = torch.log10(x_mag_ + self.eps)

        # normalize log stft
        x_mag = self.layer_norm_11(x_mag)

        # (batch, frames, bins) -> (frames, batch, bins)
        # e.g. (1, 257, 1872) -> (1872, 1, 257)
        # lstm will take (seq, batch, features) as input
        x_mag = x_mag.permute(1, 0, 2)


        ######## START OF FIRST SEPARATION CORE ########

        # (frames, batch, bins) -> (frames, batch, features)
        # e.g. (1872, 1, 257) -> (1872, 1, 128)
        x, (h0_12, c0_12) = self.lstm_11(x_mag)

        # dropout between lstm layers
        x = self.dropout_11(x)

        # (frames, batch, features) -> (frames, batch, features) 
        # e.g. (1872, 1, 128) -> (1872, 1, 128)
        x, _ = self.lstm_12(x, (h0_12, c0_12))

        # (frames, batch, features) -> (batch, frames, features)
        # e.g. (1872, 1, 128) -> (1, 1872, 128)
        x = x.permute(1, 0, 2)

        # (batch, frames, features) -> (batch, frames, bins)
        # e.g. (1, 1872, 128) -> (1, 1872, 257)
        x = self.fc_11(x)

        # (batch, frames, bins) -> (batch, frames, mask)
        # e.g. (1, 1872, 257) -> (1, 1872, 257)
        # NOTE: This is the mask
        x = self.afn_11(x)


        ######## END OF FIRST SEPARATION CORE ########

        # (batch, frames, bins * mask) -> (batch, frames, features)
        # e.g. (1, 1872, 257) -> (1, 1872, 257)
        # NOTE: ALWAYS multiplied by x_mag BEFORE norm! Fix it, IDIOT!
        # TODO: FIX IT!!
        # x = x * x_mag.permute(1, 0, 2)
        x = x * x_mag_

        # (batch, frames, features) -> (batch, features, frames)
        # e.g. (1, 1872, 257) -> (1, 257, 1872)
        # x = x.permute(0, 2, 1)

        # permutes phase to match magnitude
        # (batch, bins, frames) -> (batch, frames, bins)
        # e.g. (1, 257, 1872) -> (1, 1872, 257)
        x_phase = x_phase.permute(0, 2, 1)

        # (batch, features, frames) -> (batch, frames, samples)
        # e.g. (1, 1872, 257) -> (1, 1872, 512)
        x = self.ifft(x, x_phase)

        # accommodate input for conv1d
        # (batch, frames, samples) -> (batch, samples, frames)
        # e.g.(1, 1872, 512) -> (1, 512, 1872)
        x = x.permute(0, 2, 1)

        # (batch, samples, frames) -> (batch, features, frames)
        # e.g. (1, 512, 1872) -> (1, 256, 1872)
        x_mask_2 = self.conv1d_21(x)

        # (batch, features, frames) -> (batch, frames, features)
        # e.g. (1, 256, 1872) -> (1, 1872, 256)
        x = x_mask_2.permute(0, 2, 1)

        # layer normalization with learnable parameters
        # (batch, frames, features) -> (batch, frames, features)
        # e.g. (1, 1872, 256) -> (1, 1872, 256)
        x = self.layer_norm_21(x)


        ######## START OF SECOND SEPARATION CORE ########
        
        # (batch, frames, features) -> (frames, batch, features)
        # e.g. (1, 1872, 256) -> (1872, 1, 256)
        x = x.permute(1, 0, 2)

        # (frames, batch, features) -> (frames, batch, features)
        # e.g. (1872, 1, 256) -> (1872, 1, 128)
        x, (h0_22, c0_22) = self.lstm_21(x)

        # dropout between lstm layers
        x = self.dropout_21(x)

        # (frames, batch, features) -> (frames, batch, features)
        # e.g. (1872, 1, 128) -> (1872, 1, 128)
        x, _ = self.lstm_22(x, (h0_22, c0_22))
        
        # (frames, batch, features) -> (batch, frames, features)
        # e.g. (1872, 1, 128) -> (1, 1872, 128)
        x = x.permute(1, 0, 2)

        # (batch, frames, features) -> (batch, frames, features)
        # e.g. (1, 1872, 128) -> (1, 1872, 256)
        x = self.fc_21(x)

        # (batch, frames, features) -> (batch, frames, mask)
        # e.g. (1, 1872, 256) -> (1, 1872, 256)
        x = self.afn_21(x)

        ######## END OF SECOND SEPARATION CORE ########


        # (batch, frames, mask) -> (batch, mask, frames)
        # e.g. (1, 1872, 256) -> (1, 256, 1872)
        x = x.permute(0, 2, 1)

        # TODO: YOU ARE MISSING A MULTIPLICATION HERE, ASSHOLE!
        x = x * x_mask_2

        # (batch, mask, frames) -> (batch, mask, frames)
        # e.g. (1, 256, 1872) -> (1, 512, 1872)
        x = self.causal_conv1d_22(x)

        # (batch, mask, frames) -> (batch, frames, mask)
        # e.g. (1, 512, 1872) -> (1, 1872, 512)
        #x = x.permute(0, 2, 1)

        # COLA NEEDS TO HAVE COLUMNS OF SAMPLES VERTICALLY ORGANIZED
        # AND TIME STEPS NEED TO BE THE HORIZONTAL DIMENSION

        # overlap add. BEWARE OF BATCH SIZE!!
        x = self.overlap_and_add(x)

        # fix batch shape to match input
        # (batch, 1, 1, samples) -> (batch, samples)
        if batch_size_special == None:
            x = torch.reshape(x, (self.batch_size, -1))
        else:
            x = torch.reshape(x, (batch_size_special, -1))

        return x

# # TODO: experimental
# class DTLNTrainingProcess(TrainingProcess):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def on_train_step(self, batch_idx, batch):
#         x, y = batch
#         x = x.to(self.device)
#         y = y.to(self.device)
#         y_pred = self.model(x)
#         loss = self.criterion(y_pred, y)
#         self.optimizer.zero_grad()
#         loss.backward()

#         if self.grad_norm_clipping is not None:
#             torch.nn.utils.clip_grad_norm_(
#                     self.model.parameters(),
#                     max_norm=self.grad_norm_clipping)

#         self.optimizer.step()
#         self.running_dict.set_value('train_loss', loss.item())

#     def on_val_step(self, batch_idx, batch):
#         x, y = batch
#         x = x.to(self.device)
#         y = y.to(self.device)
#         y_pred = self.model(x)
#         loss = self.criterion(y_pred, y)

#         self.running_dict.set_value('val_loss', loss.item())

#     def on_overfit_train_step(self, batch_idx, batch):
#         self.on_train_step(batch_idx, batch)
