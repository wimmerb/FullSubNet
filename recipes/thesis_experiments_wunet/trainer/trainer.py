import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
#from util.utils import compute_STOI, compute_PESQ
from audio_zen.metrics import STOI, WB_PESQ, SI_SDR

plt.switch_backend('agg')

import tqdm

import pdb, gc


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean) in enumerate(tqdm.tqdm(self.train_data_loader)):
            # print (i)
            # print ("mix", mixture.shape)
            # print (mixture.shape)
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture)
            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        


        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        l1_losses = np.array([])
        l2_losses = np.array([])

        custom_loss = self.validation_custom_config.get("loss", None)

        si_sdr_c_n = []
        si_sdr_c_e = []
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        

        for i, (mixture, clean, name, _) in enumerate(tqdm.tqdm(self.validation_data_loader)):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            
            name = name[0]
            padded_length = 0
            

            mixture = mixture.to(self.device)  # [1, 1, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
                


            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            # print ("enhanced shape", enhanced.shape)
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            
            
            # print ("mixture shape", mixture.shape)

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            

            # print (mixture.shape)
            # print (clean.shape)
            # print (enhanced.shape)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # # Visualize spectrogram
            # noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            # enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            # clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            # if i <= visualize_spectrogram_limit:
            #     fig, axes = plt.subplots(3, 1, figsize=(6, 6))
            #     for k, mag in enumerate([
            #         noisy_mag,
            #         enhanced_mag,
            #         clean_mag,
            #     ]):
            #         axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
            #                           f"std: {np.std(mag):.3f}, "
            #                           f"max: {np.max(mag):.3f}, "
            #                           f"min: {np.min(mag):.3f}")
            #         librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
            #     plt.tight_layout()
            #     self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)
            # #print (mixture.device(), clean.device(), enhanced.device())
            # Metric
            if custom_loss == None:
                stoi_c_n.append(STOI(clean, mixture, sr=16000))
                stoi_c_e.append(STOI(clean, enhanced, sr=16000))

                si_sdr_c_n.append(SI_SDR(clean, mixture))
                si_sdr_c_e.append(SI_SDR(clean, enhanced))

                try:
                    pesq_c_n.append(WB_PESQ(clean, mixture, sr=16000))
                    pesq_c_e.append(WB_PESQ(clean, enhanced, sr=16000))
                except:
                    print ("pesq error", len (pesq_c_e))
            else:
                assert custom_loss == None
            # print(len (pesq_c_e))
            # print (pesq_c_e[-1])

            # i = 0
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             i += 1
            #             #print(type(obj), obj.size())
            #     except:
            #         pass
            # print ("NR_OBJECTS", i)
            #pdb.set_trace()
            # print (type (clean))
            # print (type (mixture))
            l1_losses = np.append(l1_losses, np.mean(np.abs(clean-enhanced)))
            l2_losses = np.append(l2_losses, np.mean((clean-enhanced)**2))
            # print(l1_losses[-1])
            # if (l1_losses[-1] == 0.0):
            #     print (clean[:100])
            #     print (mixture[:100])

        

        

        if custom_loss != None:
            print (custom_loss, "loss used")
            if custom_loss == "l1":
                score = np.mean(l1_losses)
            elif custom_loss == "l2":
                score = np.mean(l2_losses)
            else:
                assert False
        else:
            print ("no custom loss used")
        print ("l1 loss is: ", np.mean(l1_losses))
        print ("l2 loss is: ", np.mean(l2_losses))

        stoi_mean = np.mean([x for x in stoi_c_e if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])
        stoi_mean_pass = np.mean([x for x in stoi_c_n if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])

        pesq_mean = np.mean([x for x in pesq_c_e if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])
        pesq_mean_pass = np.mean([x for x in pesq_c_n if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])

        si_sdr_mean = np.mean([x for x in si_sdr_c_e if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])
        si_sdr_mean_pass = np.mean([x for x in si_sdr_c_n if x != None and str(x) != 'nan' and x != np.nan and x != np.inf])


        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": stoi_mean_pass,
            "Clean and enhanced": stoi_mean
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": pesq_mean_pass,
            "Clean and enhanced": pesq_mean
        }, epoch)

        

        score = (stoi_mean + self._transform_pesq_range(pesq_mean)) / 2

        print("========================================")
        print("STOI mean:", stoi_mean, "reference:", stoi_mean_pass)
        print("PESQ mean:", pesq_mean, "reference:", pesq_mean_pass)
        print("SI_SDR mean:", si_sdr_mean, "reference:", si_sdr_mean_pass)
        print("========================================")

        return score
