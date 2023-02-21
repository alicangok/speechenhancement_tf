from dataset import SyntheticNoisyData
from utils import get_audio_path_list, load_noise, evaluation, save_wav
from config import *
from tqdm import tqdm
from model import SEpipeline, load_model
import numpy as np
import pysepm
import tensorflow as tf

def evaluation(clean_speech, pred_speech, sr = 16_000):
    """
    return pesq, stoi, segSNR
    """
    return round(pysepm.pesq(clean_speech, pred_speech, 16_000)[1], 2), \
           round(pysepm.stoi(clean_speech, pred_speech, 16_000), 2), \
           round(pysepm.SNRseg(clean_speech, pred_speech, 16_000), 2)

DEVICE = DEVICE_NAME_TF if tf.config.list_physical_devices('GPU') else 'CPU:0'
valid_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')[0:5]
raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
noise_path.extend(get_audio_path_list(raw_noise_path, 'npy'))
noise_path.sort()
noise_types = [x.split('/')[3].split('.')[0] for x in noise_path]
noises = load_noise(noise_path)
SE = SEpipeline(n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    target = target,
    is_train = True
)

SNR = [6]

SE = load_model(model=SE, 
              device=DEVICE,
              action='predict',
              pretrained_model_path = os.path.join(model_path, pretrain_model_name))

bce = tf.keras.losses.BinaryCrossentropy()

for inx, noise in enumerate(noise_types):
    print(noise)
    valid_dataset = SyntheticNoisyData(valid_list, BATCH_SIZE, SNR, [noises[inx]], deterministic=True, random_noise=True, shuffle=False)
    valid_bar = tqdm(valid_dataset)

    for batch in valid_bar:

        with tf.device(DEVICE):
            batch = SE.stft(batch)
            batch = SE.chunk(batch)
            batch['pred_mask'] = SE.model(tf.expand_dims(batch['x'],-1))
            batch = SE.istft(batch)

            save_wav(path= os.path.join(outdir, noise + '_' + exp_name + '.wav'), wav= batch['pred_y'].numpy(), fs= SAMPLING_RATE)
            save_wav(path= os.path.join(outdir, noise + '_noisy.wav'), wav= batch['mixed_y'].numpy(), fs= SAMPLING_RATE)
            print(batch['true_y'].numpy().shape, batch['pred_y'].numpy().shape)
            print(f'recon of SE', {evaluation(batch['true_y'].numpy(), batch['pred_y'].numpy())})
            print(f'noisy', {evaluation(np.squeeze(batch['true_y'].numpy()), batch['mixed_y'].numpy())})

