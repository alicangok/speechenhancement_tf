from tensorflow.keras.utils import Sequence
from utils import load_audio, snr_mixer2
import numpy as np

crop = 2000000#99_968 # This was needed when Lambda gave OOM error

class SyntheticNoisyData(Sequence):
    
    def __init__(self, audio_list, batch_size, snr, noise, deterministic = False, random_noise= False, shuffle=True):
        super(SyntheticNoisyData, self).__init__()
        self.audios = audio_list
        self.batch_size = batch_size
        self.snr = snr
        self.noise = noise
        self.deterministic = deterministic
        self.random_noise = random_noise
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_begin(self):
        np.random.seed()

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.audios)
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.audios) // self.batch_size
    
    def __getitem__(self, idx):
        # currently assumes batch_size is 1, but it could be made able to handle any batch size
        # Generate one batch of data 
        # Generate indices of the batch

        if self.deterministic: # used for validation
            np.random.seed(idx) # by assigning the same seed for the same examples, we get the same noise for the each example in each epoch
        
        clean = load_audio(path = self.audios[idx])
        clean = clean[:min(clean.shape[0], crop)]
        
        if self.random_noise:
            noise_type = np.random.randint(0, len(self.noise))
            noise_snr = np.random.choice(self.snr)
        
        noise_start = np.random.randint(0, self.noise[noise_type].shape[0] - clean.shape[0] + 1)
        noise_snippet = self.noise[noise_type][noise_start : noise_start + clean.shape[0]]
        mixed, noise = snr_mixer2(clean, noise_snippet, noise_snr)
        
        dt = {'mixed': mixed, 'clean': clean, 'noise': noise, 'label':noise_type}
        return dt
        
