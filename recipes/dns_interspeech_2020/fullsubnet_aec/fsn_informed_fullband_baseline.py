import torch
from torch.nn import functional

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel


class Model(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_output_activate_function,
                 fb_model_hidden_size,
                 num_groups_in_drop_band,
                 norm_type="offline_laplace_norm",
                 weight_init=True,
                 variation=None,
                 
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=2*num_freqs,
            output_size=2*num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        #self.sb_num_neighbors = sb_num_neighbors
        #self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        self.variation=variation

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag, bgm_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        
        assert noisy_mag.shape == bgm_mag.shape #when this is assured, we can check derive the below info just from noisy_mag
        assert noisy_mag.dim() == 4
        bgm_mag = functional.pad(bgm_mag, [0, self.look_ahead])  # Pad the look ahead
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        bgm_mag_norm = self.norm(bgm_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        noisy_mag_norm = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)

        #[B, F*2, T]
        fb_input = torch.cat([bgm_mag_norm, noisy_mag_norm], dim=1)

        #[B, F*2, T] -> [B, 2, F, T] -> [B, F, 2, T]
        #sb_mask = self.fb_model(fb_input).reshape(batch_size, 2, num_freqs, num_frames).permute(0, 2, 1, 3).contiguous()
        if self.variation == "alternative_reshape":
            sb_mask = self.fb_model(fb_input).reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()
        else:
            sb_mask = self.fb_model(fb_input).reshape(batch_size, 2, num_freqs, num_frames).contiguous()
        #print(sb_mask.shape)
        output = sb_mask[:, :, :, self.look_ahead:]
        return output




if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        model = Model(
            sb_num_neighbors=15,
            fb_num_neighbors=0,
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_output_activate_function="ReLU",
            sb_output_activate_function=None,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            weight_init=False,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
        )
        # ipt = torch.rand(3, 800)  # 1.6s
        # ipt_len = ipt.shape[-1]
        # # 1000 frames (16s) - 5.65s (35.31%，纯模型) - 5.78s
        # # 500 frames (8s) - 3.05s (38.12%，纯模型) - 3.04s
        # # 200 frames (3.2s) - 1.19s (37.19%，纯模型) - 1.20s
        # # 100 frames (1.6s) - 0.62s (38.75%，纯模型) - 0.65s
        # start = datetime.datetime.now()
        #
        # complex_tensor = torch.stft(ipt, n_fft=512, hop_length=256)
        # mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0).unsqueeze(1)
        # print(f"STFT: {datetime.datetime.now() - start}, {mag.shape}")
        #
        # enhanced_complex_tensor = model(mag).detach().permute(0, 2, 3, 1)
        # print(enhanced_complex_tensor.shape)
        # print(f"Model Inference: {datetime.datetime.now() - start}")
        #
        # enhanced = torch.istft(enhanced_complex_tensor, 512, 256, length=ipt_len)
        # print(f"iSTFT: {datetime.datetime.now() - start}")
        #
        # print(f"{datetime.datetime.now() - start}")
        ipt = torch.rand(3, 1, 257, 200)
        print(model(ipt).shape)
