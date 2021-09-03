import os
import torch
import glob
import warnings
import soundfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .training_process import TrainingProcessCallback

# ignore matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ScheduledCheckpointCallback(TrainingProcessCallback):
    def __init__(self,
            epoch_interval: int = 25,
            checkpoint_prefix: str = 'scheduled_',
            start_epoch: int = 0,
            # TODO: to be implemented for overfit
            save_last_epoch: bool = True,
            verbose: bool = False):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.checkpoint_prefix = checkpoint_prefix
        self.start_epoch = start_epoch
        self.save_last_epoch = save_last_epoch
        self._ts_fmt = '%Y-%m-%d %H:%M:%S'
        self.verbose = verbose

    def on_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch

        # epochs are 0-indexed
        if (
            self.start_epoch > 0 and 
            current_epoch < self.start_epoch
            ):
            return
        elif (current_epoch + 1) % self.epoch_interval == 0:
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )
            
            if self.verbose:
                ts = datetime.now().strftime(self._ts_fmt)
                print(f'[{ts}] Scheduled checkpoint saved to '
                      f'{checkpoint_path}')

    def on_overfit_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch

        if (current_epoch + 1) == training_process.max_epochs:
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )

        if self.verbose:
            ts = datetime.now().strftime(self._ts_fmt)
            print(f'[{ts}] Last epoch checkpoint saved to '
                  f'{checkpoint_path}')


class BestCheckpointCallback(TrainingProcessCallback):
    def __init__(self,
                 n_best: int = 3,
                 checkpoint_prefix: str = 'best_',
                 metric: str = 'avg_val_loss',
                 direction: str = 'min',
                 verbose: bool = False):
        super().__init__()
        self.n_best = n_best
        self._session_history = []  # keeps (metric, filepath) tuples
        self.checkpoint_prefix = checkpoint_prefix
        self.metric = metric
        self.direction = direction
        self._ts_fmt = '%Y-%m-%d %H:%M:%S'
        self.verbose = verbose

    def _evaluate_metric(self, metric):
        if len(self._session_history) < self.n_best:
            return True

        session_metrics = [
                checkpoint_data[0] for checkpoint_data in self._session_history
                ]

        if self.direction == 'min' and all(
                session_metric > metric for session_metric in session_metrics
                ):
            return True

        elif self.direction == 'max' and all(
                session_metric > metric for session_metric in session_metrics
                ):
            raise NotImplementedError
        else:
            return False

    def on_val_epoch_end(self, training_process):
        # get metric
        metric = training_process.running_dict.get_last_value('avg_val_loss')
        epoch = training_process.current_epoch

        if self._evaluate_metric(metric):
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )

            # log to terminal
            if self.verbose:
                ts = datetime.now().strftime(self._ts_fmt)
                print(f'[{ts}] Best checkpoint saved to '
                      f'{checkpoint_path}')

            # most recent element always goes at the beginning
            self._session_history.insert(0, (metric, checkpoint_path))

            if len(self._session_history) > self.n_best:
                checkpoint_file_to_delete = self._session_history[-1][1]
                os.remove(checkpoint_file_to_delete)
                del self._session_history[-1]

                if self.verbose:
                    ts = datetime.now().strftime(self._ts_fmt)
                    print(f'[{ts}] Old best checkpoint deleted from '
                          f'{checkpoint_file_to_delete}')


# TODO: Do a generalize object that can take i/o from different types such as
# spectrogram, complex input, only audio data, etc
class AudioProcessTrackerCallback(TrainingProcessCallback):
    def __init__(self,
                 input_dir: str = None,
                 output_dir: str = None,
                 epoch_interval: int = 25,
                 overfit_epoch_interval: int = 500, # to be implemented
                 sample_original_files: bool = True,
                 sample_rate: int = 16000,
                 sample_duration: int = 10,
                 log_audio_file: bool = True,
                 log_waveform_fig: bool = True,
                 log_linspec_fig: bool = True,
                 log_logspec_fig: bool = True,
                 window_size: int = 320,
                 hop_size: int = 160,
                 # TODO: to be replaced by util or union with callable
                 pred_fn: str = 'std_pred_fn'):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.epoch_interval = epoch_interval
        self.sample_original_files = sample_original_files
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.log_audio_file = log_audio_file
        self.log_waveform_fig = log_waveform_fig
        self.log_linspec_fig = log_linspec_fig
        self.log_logspec_fig = log_logspec_fig
        self.window_size = window_size
        self.hop_size = hop_size

        # TODO: fix to accept only callable
        if callable(pred_fn):
            self.pred_fn = pred_fn
        else:
            self.pred_fn = getattr(self, pred_fn)

        self.input_files, self.output_files = self._collect_files()

    def std_pred_fn(self, audio_data: torch.tensor, training_process):
        pred_audio = training_process.model(audio_data)
        pred_audio = pred_audio.cpu().numpy().reshape(-1)

        return pred_audio

    def cruse_pred_fn(self, audio_data: torch.tensor, training_process):
        """ Must return the predicted audio ready to be plotted """
        # calculate complex spectrum and log pow spec (lps)

        # TODO: avoid hardcoding window
        hann_window = torch.hann_window(self.window_size).to(audio_data.device)

        # TODO: fixed transformation for now, allow flexibility
        # or allow just external callables
        audio_complex = torch.stft(audio_data,
                                   onesided=True,
                                   n_fft=self.window_size,
                                   center=True,
                                   hop_length=self.hop_size,
                                   normalized=False,
                                   window=hann_window,
                                   return_complex=True)

        audio_lps = torch.log10(torch.abs(audio_complex) ** 2 + 1e-7)
        pred_audio_mask = training_process.model(audio_lps)
        pred_audio_complex = (
                pred_audio_mask.squeeze(1).permute(0, 2, 1) * audio_complex
                )

        pred_audio = torch.istft(pred_audio_complex,
                                 onesided=True,
                                 n_fft=self.window_size,
                                 center=True,
                                 hop_length=self.hop_size,
                                 normalized=False,
                                 window=hann_window)

        pred_audio = pred_audio.cpu().numpy().reshape(-1)

        return pred_audio

    def _collect_files(self, ext='*.wav'):
        # assumes equal names on input and output
        # TODO: allow key callback to sort files
        input_files = glob.glob(os.path.join(self.input_dir, ext))
        output_files = glob.glob(os.path.join(self.output_dir, ext))
        return sorted(input_files), sorted(output_files)

    def _log_audio_file(self, logger, tag, audio_data, epoch, sample_rate):
        logger.add_audio(tag, audio_data, epoch, sample_rate)

    def _log_waveform_fig(self, logger, tag, audio_data, epoch, sample_rate):
        t_ax = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

        fig, ax = plt.subplots()
        ax.set_title(tag)
        ax.set_xlim([0, len(audio_data) / sample_rate])
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.plot(t_ax, audio_data)

        logger.add_figure(tag, fig, epoch)

    def _log_spectrogram_fig(self, logger, tag, audio_data,
                             epoch, sample_rate, 
                             window_size=2048, hop_size=512,
                             x_axis='time', y_axis='linear', fmt='%+.2d dB'):
        audio_data_stft = librosa.stft(audio_data,
                                       n_fft=window_size,
                                       hop_length=hop_size)
        audio_data_stft_db = librosa.amplitude_to_db(np.abs(audio_data_stft),
                                                     ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(audio_data_stft_db,
                                       x_axis=x_axis,
                                       y_axis=y_axis,
                                       ax=ax,
                                       sr=sample_rate,
                                       hop_length=hop_size)
        ax.set_title(tag)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=ax, format=fmt)
        logger.add_figure(tag, fig, epoch)

    def on_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch
        logger = training_process.logger

        if current_epoch == 0:
            for idx, input_file in enumerate(self.input_files):
                audio_data, _ = soundfile.read(
                        input_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype='float32'
                        )

                if self.log_audio_file:
                    self._log_audio_file(logger, 
                                    f'Initial/source_{idx}',
                                    audio_data,
                                    current_epoch,
                                    self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f'Initial/source_{idx}.waveform',
                                           audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f'Initial/source_{idx}.linspec',
                                             audio_data,
                                             current_epoch,
                                             self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f'Initial/source_{idx}.logspec',
                                             audio_data,
                                             current_epoch,
                                             self.sample_rate,
                                             y_axis='log')

            for idx, output_file in enumerate(self.output_files):
                audio_data, _ = soundfile.read(
                        output_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype='float32'
                        )

                if self.log_audio_file:
                    self._log_audio_file(logger,
                                    f'Initial/target_{idx}',
                                    audio_data,
                                    current_epoch,
                                    self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f'Initial/target_{idx}.waveform',
                                           audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f'Initial/target_{idx}.linspec',
                                              audio_data,
                                              current_epoch,
                                              self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f'Initial/target_{idx}.logspec',
                                              audio_data,
                                              current_epoch,
                                              self.sample_rate,
                                              y_axis='log')

        if (current_epoch == 0 or
            (current_epoch + 1) % self.epoch_interval == 0):
            
            for idx, input_file in enumerate(self.input_files):
                audio_data, _ = soundfile.read(
                        input_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype='float32'
                        )
                
                # add reshaping to external process
                audio_data = audio_data.reshape(1, -1)
                audio_data = torch.from_numpy(audio_data) \
                                  .to(training_process.device)

                if callable(self.pred_fn):
                    pred_audio_data = self.pred_fn(audio_data)
                elif self.pred_fn is not None:
                    pred_audio_data = self.pred_fn(audio_data, training_process)
                else:
                    raise NotImplementedError(
                            "A prediction function is required"
                            )

                if self.log_audio_file:
                    self._log_audio_file(logger,
                                         f'Predicted/result_{idx}',
                                         pred_audio_data,
                                         current_epoch,
                                         self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f'Predicted/result_{idx}.waveform',
                                           pred_audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f'Predicted/result_{idx}.linspec',
                                             pred_audio_data,
                                             current_epoch,
                                             self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f'Predicted/result_{idx}.logspec',
                                              pred_audio_data,
                                              current_epoch,
                                              self.sample_rate,
                                              y_axis='log')

    def on_overfit_val_epoch_end(self, training_process):
        self.on_val_epoch_end(training_process)


# TODO: to be implemented
class SlackNotificationCallback(TrainingProcessCallback):
    def __init__():
        super().__init__()
