import os
import time
import json
import glob
import pesq
import subprocess
import soundfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from prettytable import PrettyTable
from pystoi.stoi import stoi
from typing import List
from urllib.parse import urlparse, urljoin


# TODO: Move it to a common place to avoid repeating code since it is also
# used in other classes (e.g. dataset)
def get_fileid_n(filename: str):
    filename = os.path.splitext(filename)[0]
    file_id = filename.split('_fileid_')[1].zfill(10)
    return file_id

# TODO: Same as for get_fileid_n
def collect_wav_files(input_dir, key=get_fileid_n):
    return sorted(glob.glob(os.path.join(input_dir, '*.wav')), key=key)

# TODO: Replace later on by logger
def log(msg, ts_fmt='%Y-%m-%d %H:%M:%S'):
    ts = datetime.now().strftime(ts_fmt)
    print(f'[{ts}] {msg}')

def run_si_sdr(reference, estimation):
    """ Scale-Invatiant Signal-to-Distortion Ratio (SI-SDR) 

    The current implementation details can be found in this paper: 
    http://www.merl.com/publications/docs/TR2019-013.pdf

    Args:
        reference (numpy.ndarray): Reference audio (clean speech)
        estimation (numpy.ndarray): Predicted speech
    Returns:
        si_sdr (float32): SI-SDR
    """
    # TODO: Replace it from the original implementation to simply raise an
    # exception if the arrays need some reshaping
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # Alpha after Equation (3) in the paper
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
                      / reference_energy

    # e_target in Equation (4)
    projection = optimal_scaling * reference

    # e_res in Equation (4)
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)

    return 10 * np.log10(ratio)

def run_stoi(reference, estimation, sample_rate):
    """ Wrapper to allow independent axis for STOI 
    Args:
        reference (numpy.ndarray): Reference audio (clean speech)
        estimation (numpy.ndarray): Predicted speech
        sample_rate (int): Sample rate of the audio data
    """
    # TODO: Replace it from the original implementation to simply raise an 
    # exception if the arrays need some reshaping
    estimation, reference = np.broadcast_arrays(estimation, reference)

    if reference.ndim >= 2:
        return np.array([
                stoi(x_entry, y_entry, sample_rate=sample_rate)
                for x_entry, y_entry in zip(reference, estimation)
            ])
    else:
        return stoi(reference, estimation, fs_sig=sample_rate)

def run_pesq(reference, estimation, sample_rate):
    return pesq.pesq(sample_rate, reference, estimation, 'wb')

def run_visqol(reference_audio_data, estimation_audio_data, 
                sample_rate, tmp_dir):
    # temporary file names
    tmp_reference = datetime.now().strftime('%Y%m%d%H%M%S_ref.wav')
    tmp_estimation = datetime.now().strftime('%Y%m%d%H%M%S_est.wav')

    soundfile.write(
            os.path.join(tmp_dir, tmp_reference), 
            reference_audio_data, 
            sample_rate
            )

    soundfile.write(
            os.path.join(tmp_dir, tmp_estimation), 
            estimation_audio_data, 
            sample_rate
            )

    # this assumes visqol is installed as a command line tool already
    reference_abs_path = os.path.abspath(
            os.path.join(tmp_dir, tmp_reference)
            )
    estimation_abs_path = os.path.abspath(
            os.path.join(tmp_dir, tmp_estimation)
            )

    visqol_cmd = ("cd visqol; ./bazel-bin/visqol "
                 f"--reference_file {reference_abs_path} "
                 f"--degraded_file {estimation_abs_path} "
                 f"--use_speech_mode")

    visqol = subprocess.run(visqol_cmd, shell=True, 
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # parse stdout to get the current float value
    visqol = visqol.stdout.decode('utf-8').split('\t')[-1].replace('\n', '')
    visqol = float(visqol)

    # remove files to avoid filling space storage
    os.remove(reference_abs_path)
    os.remove(estimation_abs_path)

    return visqol

def run_dnsmos(estimation_audio_data):
    # TODO: Move to a class or somewhere else to avoid request the key
    # every time. Also, made key private when the repo goes public
    SCORING_URI = 'https://dnsmos-2.azurewebsites.net/score'
    AUTH_KEY = 'dXBmLWVkdTpkbnNtb3M='
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = f'Basic {AUTH_KEY }'

    data = {"data": estimation_audio_data.tolist()}
    input_data = json.dumps(data)

    u = urlparse(SCORING_URI)
    response = requests.post(urljoin("https://" + u.netloc, "score"),
                             data=input_data,
                             headers=headers)
    return response.json()['mos']

def save_score_report(scores: List[tuple], filename='', mode='w', header=True):
    # NOTE: mode='a' will append to existing file, use header=False in such case
    df = pd.DataFrame(dict(scores))
    df.to_csv(os.path.join('reports', filename), index=False,
            mode=mode, header=header)

def save_score_summary(scores: List[tuple], filename='', 
        print_result=True, save_result=False):
    summary_table = PrettyTable()

    col_names = ['stats']

    for score in scores:
        col_names.append(score[0])

    summary_table.add_column('stats', [
        'min', 'max', 'average', 'stdev', '20%', '40%', 'median', '60%', '80%'
        ])

    for score in scores:
        score_min = np.min(score[1])
        score_max = np.max(score[1])
        score_avg = np.mean(score[1])
        score_stdev = np.std(score[1])
        score_20 = np.percentile(score[1], 20)
        score_40 = np.percentile(score[1], 40)
        score_median = np.median(score[1])
        score_60 = np.percentile(score[1], 60)
        score_80 = np.percentile(score[1], 80)

        summary_table.add_column(score[0], [
            f'{score_min:.4f}',
            f'{score_max:.4f}',
            f'{score_avg:.4f}',
            f'{score_stdev:.4f}',
            f'{score_20:.4f}',
            f'{score_40:.4f}',
            f'{score_median:.4f}',
            f'{score_60:.4f}',
            f'{score_80:.4f}'
            ])

    if print_result:
        print(summary_table)

    if save_result: 
        with open(os.path.join('reports', filename), 'w') as f:
                f.write(summary_table.get_string())
