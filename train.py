import os
from utils import get_audio_path_list, load_noise
from dataset import SyntheticNoisyData
from config import *
from model import SEpipeline
from trainer import Trainer
import tensorflow as tf

DEVICE = DEVICE_NAME_TF if tf.config.list_physical_devices('GPU') else 'CPU:0'

print(os.getcwd())
if not os.path.exists(model_path):
    os.mkdir(model_path)

train_list = get_audio_path_list(os.path.join(DATADIR, 'train'), 'flac')
valid_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
noise_path.extend(get_audio_path_list(raw_noise_path, 'npy'))
noise_path.sort()
noises = load_noise(noise_path)

noise_dataset = SyntheticNoisyData(train_list, BATCH_SIZE, SNR, noises, random_noise=True)
valid_dataset = SyntheticNoisyData(valid_list, BATCH_SIZE, SNR, noises, deterministic=True, random_noise=True, shuffle=False) # add [0:50] to end of valid_list if you don't want to use all the validation data for a quicker evaluation

SE = SEpipeline(n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    learning_rate=lr,
    target = target,
    is_train = True
)

# not implemented yet
# if action == 'retrain':
# epoch, SE, optimizer = load_model(model=SE, 
#                                   optimizer=optimizer,
#                                   action=action,
#                                   pretrained_model_path=os.path.join(model_path, pretrain_model_name))

trainer = Trainer(model=SE,
                  train_loader= noise_dataset, 
                  valid_loader= valid_dataset, 
                  device= DEVICE)

trainer.train(epoch_to_start_from=0,
              total_epochs=EPOCH,
              model_path = model_path)

print('Done Training')
