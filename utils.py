import os, glob
from tqdm import tqdm 
import librosa
import numpy as np
import soundfile as sf
import pysepm

def get_audio_path_list(dir, ext):
    """
    >>> train_path = os.path.join(root, 'train')
    >>> train_list = get_audio_path_list(train_path, 'flac')
    return audio_path_list
    """
    wav_path_list = []
    wav_path_list.extend(glob.glob(os.path.join(dir, f"**/*.{ext}"), recursive=True))
    return wav_path_list

def load_noise(noise_paths):
    print('loading noise')
    return {i : np.load(path) for i, path in tqdm(enumerate(noise_paths))}

def load_audio(path, sr= 16000):
    # return numpy array
    waveform, sr = librosa.load(path, sr=sr)
    return waveform

def save_wav(path, wav, fs):
    """
    Save .wav file.
    Argument/s:
        path - absolute path to save .wav file.
        wav - waveform to be saved.
        fs - sampling frequency.
    """
    wav = np.squeeze(wav)
    sf.write(path, wav, fs)

def snr_mixer2(clean, noise, snr):
    eps= 1e-8
    s_pwr = np.var(clean)
    noise = noise - np.mean(noise)
    n_var = s_pwr / (10**(snr / 10))
    noise = np.sqrt(n_var) * noise / (np.std(noise) + eps)
    noisyspeech = clean + noise

    if max(abs(noisyspeech)) > 1:
        noisyspeech /= max(abs(noisyspeech))
    #valid = True
    #return valid, noisyspeech, noise
    return noisyspeech, noise

def irm(clean_mag, noise_mag):
    # ideal ratio mask, to recover: predicted mask * noisy mag = clean mag
    eps= 1e-8
    return (clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2 + eps)) ** 0.5

def evaluation(clean_speech, pred_speech, sr = 16_000):
    # return pesq, stoi, segSNR
    return round(pysepm.pesq(clean_speech[:pred_speech.shape[0]], pred_speech, 16_000)[1], 2), \
           round(pysepm.stoi(clean_speech[:pred_speech.shape[0]], pred_speech, 16_000), 2), \
           round(pysepm.SNRseg(clean_speech[:pred_speech.shape[0]], pred_speech, 16_000), 2)
