import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

from audio_zen.acoustics.feature import mag_phase, drop_band, save_wav
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from audio_zen.trainer.base_trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, dist, rank, config, resume, only_validation, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super().__init__(dist, rank, config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):

        loss_total = 0.0
        progress_bar = None

        if self.rank == 0:
            #print (len (self.train_dataloader))
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Training")
        
        # i = 0

        for noisy, clean, bgm in self.train_dataloader:
            # if i > 100:
            #     break
            # i += 1

            self.optimizer.zero_grad()
            # print ("NEW BATCH")
            # print (noisy.shape)

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)
            bgm = bgm.to(self.rank)
            
            # (16, 49152 or else)

            assert torch.isfinite(clean).all(), "clean not finite!"
            assert torch.isfinite(noisy).all(), "noisy not finite!"
            assert torch.isfinite(bgm).all(), "bgm not finite!"

            with autocast(enabled=self.use_amp):
                pred = self.model(noisy, bgm)
                loss = self.loss_function(clean, pred)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

            if self.rank == 0:
                progress_bar.update(1)

        if self.rank == 0:
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(total=len(self.valid_dataloader), desc=f"Validation")

        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"all": 0.0}
        item_idx_list = {"all": 0.0}
        noisy_y_list = {"all": []}
        clean_y_list = {"all": []}
        enhanced_y_list = {"all": []}
        validation_score_list = {"all": 0.0}


        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, bgm, name, speech_type) in enumerate(self.valid_dataloader):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]


            if not (np.isfinite(noisy).all() and np.isfinite(clean).all()):
                assert False

            # eps=1e-6
            # bla[np.abs(bla) < 0.1] = 0.25
            #print (noisy.shape, clean.shape)
            
            # with torch.no_grad():
            #     print(torch.sum(torch.abs(clean)))

            #print(speech_type)

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)
            bgm = bgm.to(self.rank)
            
            # (16, 49152 or else)

            assert torch.isfinite(clean).all(), "clean not finite!"
            assert torch.isfinite(noisy).all(), "noisy not finite!"
            assert torch.isfinite(bgm).all(), "bgm not finite!"

            pred = self.model(noisy, bgm)
            loss = self.loss_function(clean, pred)
            enhanced = pred

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            if not np.isfinite(enhanced).all():
                print ("enhanced seems problematic...")
                assert False

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            

            if item_idx_list[speech_type] <= visualization_n_samples:
                if not np.isfinite(noisy).all():
                    assert False
                if not np.isfinite(clean).all():
                    assert False
                if not np.isfinite(enhanced).all():
                    print (enhanced[np.abs(enhanced) > 1000])
                    print(np.mean(clean))
                    print(np.mean(noisy))
                    print(np.mean(enhanced))
                    print(np.mean(clean))
                    print (np.max(np.abs(enhanced)))
                    print (enhanced)
                    continue
                    assert False
                print("executing spec_audio_visualization")
                print (enhanced)
                
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)
            # else:
            #     print("GOING TO spec_audio_visualization, but not executing")
            #     self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)



            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

            if self.rank == 0:
                progress_bar.update(1)

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ["all"]:
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            for i in range(5):
                print("saving audio...")
                tmp_name = f"{i}.wav"
                save_wav (f"tmp/noisy_{tmp_name}", noisy_y_list[speech_type][i])
                save_wav (f"tmp/enhanced_{tmp_name}", enhanced_y_list[speech_type][i])
                save_wav (f"tmp/clean_{tmp_name}", clean_y_list[speech_type][i])

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

            

        return validation_score_list["all"]
