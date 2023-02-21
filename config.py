import os

DATADIR = "../data"
SNR = [6, 9, 12]
SAMPLING_RATE = 16000
N_d = 512  # window duration (samples)
N_s = 128  # window shift (samples)
K = 512  # number of frequency bins
CHUNK_SIZE = 32 # temporal context 32 frames * 512 sample shift per frame / 16000 samples/sec = 0.25sec
target = 'mask'
action = 'train'
BATCH_SIZE = 1
lr = 3e-4
EPOCH = 40
pretrain_model_name = 'epoch15'
exp_name = 'tensorflow_noBN'
model_dir = "../models"
recovered_path = os.path.join(model_dir, exp_name)
model_path = os.path.join(model_dir, exp_name)
outdir = '../results'
DEVICE_NAME_TF = 'GPU:0'
