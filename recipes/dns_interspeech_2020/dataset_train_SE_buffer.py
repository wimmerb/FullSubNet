import random
import os

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, save_wav, subsample, subsample_audio_tensor
from audio_zen.utils import expand_path, basename

from itertools import product
import torch
import torchaudio
from sklearn.utils import shuffle
import traceback


class Dataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 clean_dataset_limit,
                 clean_dataset_offset,
                 noise_dataset,
                 noise_dataset_limit,
                 noise_dataset_offset,
                 rir_dataset,
                 rir_dataset_limit,
                 rir_dataset_offset,
                 snr_range,
                 reverb_proportion,
                 silence_length,
                 target_dB_FS,
                 target_dB_FS_floating_value,
                 sub_sample_length,
                 sr,
                 pre_load_clean_dataset,
                 pre_load_noise,
                 pre_load_rir,
                 num_workers,
                 num_trainfile_examples,
                 buffer_size
                 ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # debug args
        self.num_trainfile_examples = num_trainfile_examples

        # acoustics args
        self.sr = sr
        print ("init", num_workers)

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_dataset), "r")]
        noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_dataset), "r")]
        rir_dataset_list = [line.rstrip('\n') for line in open(expand_path(rir_dataset), "r")]

        clean_dataset_list = self._offset_and_limit(clean_dataset_list, clean_dataset_offset, clean_dataset_limit)
        noise_dataset_list = self._offset_and_limit(noise_dataset_list, noise_dataset_offset, noise_dataset_limit)
        rir_dataset_list = self._offset_and_limit(rir_dataset_list, rir_dataset_offset, rir_dataset_limit)

        if pre_load_clean_dataset:
            clean_dataset_list = self._preload_dataset(clean_dataset_list, remark="Clean Dataset")

        if pre_load_noise:
            noise_dataset_list = self._preload_dataset(noise_dataset_list, remark="Noise Dataset")

        if pre_load_rir:
            rir_dataset_list = self._preload_dataset(rir_dataset_list, remark="RIR Dataset")

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)

        self.buffer_size = buffer_size # PRODUCES buffersize**2 entries (see buffer_positions)
        self.buffer_offset = 0
        self.buffer_picking_item_position = 0
        self.buffer_needs_redo = True
        self.buffer_clean_tensors = []
        self.buffer_picking_order = []
        self.buffer_convoluted_speech = None
        self.buffer_noise = []

    def __len__(self):
        return self.length

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(f_path, self.sr) for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length


        noise_fns = "NOISE"

        

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)

            noise_fns += "_" + basename(noise_file)[0]
            #print ("LOADING NEW NOISE")

            #noise_new_added = load_wav(noise_file, sr=self.sr)

            noise_new_added = (self._get_sample_norm(noise_file, resample = self.sr, processed = True)[0])
            noise_new_added = np.transpose (np.array (noise_new_added))

            #print ("Noise length in minutes", len(noise_new_added)/self.sr/60, "FILE", noise_file)

            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # Adding silence between snippets of noise
            # 如果还需要添加新的噪声，就插入一个小静音段
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        # WE HANDLE THIS IN POP_ITEM NOW
        # if len(noise_y) > target_length:
        #     idx_start = np.random.randint(len(noise_y) - target_length)
        #     noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y, noise_fns

    @staticmethod
    def snr_mix(clean_y, noise_y, conv_y, snr, target_dB_FS, target_dB_FS_floating_value, use_reverb=True, eps=1e-6, num_trainfile_examples=10, sr=16000):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if use_reverb is True:
            clean_y_reverberant = conv_y
        else:
            clean_y_reverberant = clean_y

       
        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        clean_y_reverberant, _ = norm_amplitude(clean_y_reverberant)
        clean_y_reverberant, _, _ = tailor_dB_FS(clean_y_reverberant, target_dB_FS)
        clean_reverberant_rms = (clean_y_reverberant ** 2).mean() ** 0.5

        # if(rir is not None):
        #     print (clean_rms, clean_reverberant_rms)
        #     save_wav(f"reverberant_{num_trainfile_examples}.wav", clean_y_reverberant, sr)
        #     save_wav(f"clean_{num_trainfile_examples}.wav", clean_y, sr)

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_reverberant_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        #snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar


        
        noisy_y = clean_y_reverberant + noise_y
        
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar
        

        if num_trainfile_examples > 0:
            #print (np.max(noisy_y))
            #print (np.max(clean_y))

            #print("CONV RMS", (noisy_y ** 2).mean() ** 0.5)
            #print("CLEAN RMS", (clean_y ** 2).mean() ** 0.5)
            save_wav(f"noisy_{num_trainfile_examples}{'_rev' if use_reverb else ''}.wav", noisy_y, sr)
            save_wav(f"clean_{num_trainfile_examples}{'_rev' if use_reverb else ''}.wav", clean_y, sr)

        return noisy_y, clean_y

    
    def _get_sample(self, path, resample=None):
        effects = [
            ["remix", "1"]
        ]
        #effects = [[]]
        if resample:
            effects.append(["rate", f'{resample}'])
        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

    def _get_sample_norm(self, snd_path, resample=None, processed=False):
        # print ("RIR SAMPLE")
        snd_raw, sample_rate = self._get_sample(snd_path, resample=resample)
        # print (rir_raw.shape)
        if not processed:
            return snd_raw, sample_rate
        # print(rir_raw.shape)
        snd = snd_raw # [:, int(sample_rate*1.01):int(sample_rate*1.3)]
        snd = snd / torch.norm(snd, p=2)
        snd = torch.flip(snd, [1])
        # print (rir.shape)
        return snd, sample_rate


    def fill_conv_buffer(self):
        

        clean_files = []
        for index in range(self.buffer_offset, self.buffer_offset + self.buffer_size):
            index = index % self.length
            clean_files.append( self.clean_dataset_list[index])
        #print (clean_files)
        
        self.buffer_offset = (self.buffer_offset + self.buffer_size) % self.length
        #print ("BUFFER OFFSET", self.buffer_offset)
        

        self.buffer_clean_tensors = []
        for fn in clean_files:
            #clean_y = load_wav(fn, sr=self.sr)
            clean_y, sample_rate = self._get_sample_norm(fn, resample=self.sr)
            clean_y = subsample_audio_tensor (clean_y, sub_sample_length=int(np.floor(self.sub_sample_length * self.sr)))
            # print (clean_y.shape)
            self.buffer_clean_tensors.append (clean_y)

        rirs = []
        for i in range (self.buffer_size):
            rir_file = self._random_select_from(self.rir_dataset_list)
            rirs.append (self._get_sample_norm(rir_file, resample = self.sr, processed = True)[0])
            #rirs.append (load_wav(rir_file, sr=self.sr))

        # print ("DONE LOADING")
        # print ([x.shape for x in self.buffer_clean_tensors])
        # print ("XXXXXXX")
        # print ([x.shape for x in rirs])
        rirs = np.array(rirs, dtype=object)

        # print ("XXXXXXX")

        with torch.no_grad():
            max_rir_len = max([rir_sample.shape[1] for rir_sample in rirs])
            speech_ = torch.stack([torch.nn.functional.pad(clean_y, (max_rir_len-1,0)) for clean_y in self.buffer_clean_tensors.copy()]).cuda()
            rir_ = torch.stack([torch.nn.functional.pad(rir_sample, (max_rir_len-rir_sample.shape[1],0)) for rir_sample in rirs]).cuda()

            # print (speech_.device)

            # print (speech_.shape)
            # print (rir_.shape)

            speech = torch.nn.functional.conv1d(speech_, rir_)

            #print ("out...", speech.device)

            #print (self.sub_sample_length * self.sr)
            #print (speech.shape)

            self.buffer_convoluted_speech = speech.cpu()
            
            del speech
            del speech_
            del rir_
            # conv_1 = np.array(speech[1,1,:])
            # conv_2 = np.array(speech[1,2,:])

            # save_wav("conv_1.wav", conv_1, self.sr)
            # save_wav("conv_2.wav", conv_2, self.sr)

            #print ("BUFFERING NOISE...")

            self.buffer_noise = [self._select_noise_y(target_length = int(np.floor(self.sub_sample_length * self.sr)))[0] for _ in range(self.buffer_size)]

    def pop_from_buffer(self):
        

        if self.buffer_needs_redo:
            
            self.fill_conv_buffer()
            self.buffer_picking_item_position = 0
            self.buffer_picking_order = shuffle([x for x in product(range(self.buffer_size), repeat=2)])
            self.buffer_needs_redo = False

            #print ("NEW BUFFER")
            #print (self.buffer_offset)
            #print ("SAMPLE NR:", self.buffer_offset*self.buffer_size)
        
        speech_pos, rir_pos = self.buffer_picking_order[self.buffer_picking_item_position]

        clean_y = np.transpose (np.array (self.buffer_clean_tensors[speech_pos]))[:,0]
        #print ("CLEAN SHAPE", clean_y.shape)
        conv_y = np.transpose (np.array (self.buffer_convoluted_speech [speech_pos, rir_pos, :]))
        #print ("CONV SHAPE", conv_y.shape)
        assert clean_y.shape[0] == conv_y.shape[0]


        self.buffer_picking_item_position += 1
        if self.buffer_picking_item_position >= len(self.buffer_picking_order):
            self.buffer_needs_redo = True

        noise_y = random.choice(self.buffer_noise)

        target_length = int(np.floor(self.sub_sample_length * self.sr))
        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        # save_wav(f"{speech_pos}_{rir_pos}_clean.wav", clean_y, self.sr)
        # save_wav(f"{speech_pos}_{rir_pos}_conv.wav", conv_y, self.sr)
        

        return clean_y, conv_y, noise_y

    def __getitem__(self, item):
        # print ("*******************************")
        # print ("*******************************")
        # print ("*******************************")
        # for line in traceback.format_stack():
        #     print(line.strip())
        # print ("*******************************")
        # print ("*******************************")
        # print ("*******************************")
        #print (self.buffer_picking_item_position)
        #
        

        #get random item from buffer
        clean_y, conv_y, noise_y = self.pop_from_buffer()


        # clean_file = self.clean_dataset_list[item]
        # clean_y = load_wav(clean_file, sr=self.sr)
        # clean_y = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr))
        

        #noise_y, noise_fns = self._select_noise_y(target_length=len(clean_y))
        assert len(clean_y) == len(noise_y), f"Inequality: {len(clean_y)} {len(noise_y)}"

        snr = self._random_select_from(self.snr_list)
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)

        
        # rir = None
        # if use_reverb:
        #     rir_file = self._random_select_from(self.rir_dataset_list)
        #     rir = load_wav(rir_file, sr=self.sr)
        
        # # rir = load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr) if use_reverb else None

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            conv_y=conv_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            use_reverb=use_reverb,
            num_trainfile_examples=self.num_trainfile_examples,
            sr=self.sr
        )
        self.num_trainfile_examples-=1

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        #TODO MORE SOPHISTICATED
        fid = item

        # #save noisy file
        # parent_dir = "../VALIDATION_SET_3/noisy/no_reverb"
        # filename = "CLEAN_" + basename(clean_file)[0] + noise_fns + ".wav"
        # if use_reverb:
        #     parent_dir = "../VALIDATION_SET_3/noisy/with_reverb"
        #     filename = "CLEAN_" + basename(clean_file)[0] + noise_fns + "_RIR_" + basename(rir_file)[0] + ".wav"
        # #save_wav(os.path.join(parent_dir, filename), noisy_y, self.sr)
        
        # #save clean file
        # parent_dir = "../VALIDATION_SET_3/clean"
        # filename = "CLEAN_" + basename(clean_file)[0] + ".wav"
        # #save_wav(os.path.join(parent_dir, filename), clean_y, self.sr)
        

        return noisy_y, clean_y

