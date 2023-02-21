from utils import irm
from senet_tf import SENet
import tensorflow as tf
from tensorflow.signal import stft, inverse_stft, hann_window, inverse_stft_window_fn
from tensorflow.keras.models import Model

class SEpipeline(Model):

    def __init__(self, n_fft, hop_len, win_len, device, chunk_size, learning_rate=3e-4, criterion='binary_crossentropy', **kwargs):
        super(SEpipeline, self).__init__()
        self.chunk = ChunkDatav3(chunk_size= chunk_size, target= kwargs['target'], train=kwargs['is_train'], device=device)
        with tf.device(device):
            model = SENet()
        self.stft = tf_stft(n_fft=n_fft, 
                           hop_length=hop_len, 
                           win_length=win_len, 
                           device=device, 
                           train=kwargs['is_train'])
        self.istft = tf_istft(n_fft =n_fft, 
                             hop_length=hop_len,
                             win_length=win_len,
                             device=device,
                             chunk_size=chunk_size,
                             target= kwargs['target'],
                             train=kwargs['is_train'])
        self.model = model
        self.is_train = kwargs['is_train']

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        bce = tf.keras.losses.BinaryCrossentropy()
        with tf.device(device):
            self.model.compile(loss=bce, optimizer=opt, metrics=["binary_accuracy"])


    def call(self, dt, recon=False):
        dt = self.stft(dt)
        dt = self.chunk(dt)
        dt = self.model(dt['x'])
        if recon:
            dt = self.istft(dt)
        return dt

class tf_stft(Model):
    
    def __init__(self, n_fft, hop_length, win_length, device, train):
        super(tf_stft, self).__init__()
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.device = device
        self.train = train
        self.window = hann_window
    
    def call(self, dt):
        with tf.device(self.device):
            #key_list = list(dt.keys())
            key_list = ['mixed', 'clean', 'noise']
            for key in key_list:
                fft = stft(dt[key], frame_length=self.win_length, frame_step=self.hop_length, fft_length=self.n_fft, window_fn=hann_window)
                mag = tf.abs(fft)
                if key == 'mixed':
                    dt['phase'] = tf.exp(tf.complex(0.,tf.math.angle(fft)))
                dt[f'{key}_mag'] = mag
                dt[f'{key}_mag_comp'] = tf.math.log1p(mag)
                dt[f'{key}_mag_comp'] = tf.math.minimum(dt[f'{key}_mag_comp'], 2.0)
           
        if self.train:        
            mask = irm(dt['clean_mag'], dt['noise_mag'])
            dt['mask'] = mask > 0.5
        
        return dt
    
class tf_istft(Model):
    
    def __init__(self, n_fft, hop_length, win_length, chunk_size, device, target, train):
        super(tf_istft, self).__init__()
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.device = device
        self.chunk_size = chunk_size
        self.target = target
        self.train = train
        self.istft_window = inverse_stft_window_fn(frame_step=self.win_length, forward_window_fn = hann_window)
    
    def mask_recover(self, dt):
        mask_shape = list(dt['pred_mask'].shape)
        freq = mask_shape[-1]
        pred_mask = dt['pred_mask']

        if freq == 256:
            pred_mask = tf.pad(pred_mask, [[0,0],[0,0],[0,1]])
            freq += 1
            mask_shape[-1] = freq
            pred_mask = tf.reshape(pred_mask, [-1, freq])
        else:
            pred_mask = tf.reshape(pred_mask, [-1, freq])
        dt['pred_y'] = pred_mask * dt['mixed_mag'][:pred_mask.shape[0]]
        dt['pred_y'] = tf.reshape(dt['pred_y'], mask_shape)
        
        return dt
    
    
    def cnn2d_recover(self, dt):
        dt['pred_y'] = tf.reshape(dt['pred_y'],[-1, dt['pred_y'].shape[-1]])
        lens = dt['phase'].shape[0]
        dt['pred_y'] = tf.math.multiply(tf.cast(dt['pred_y'][:lens], tf.complex64), dt['phase'][:dt['pred_y'][:lens].shape[0]])
        if self.train or 'clean_mag' in dt.keys():
            dt['true_y'] = tf.math.multiply(tf.cast(dt['clean_mag'][:lens], tf.complex64), dt['phase'])
        dt['mixed_y'] = tf.math.multiply(tf.cast(dt['mixed_mag'][:lens], tf.complex64), dt['phase'])
        return dt
    
    def call(self, dt):
        
        if self.target == 'mask':  
            dt = self.mask_recover(dt)
        else:
            if 'pred_mask' in dt:
                dt['pred_y'] = dt['pred_mask']
                    
        dt = self.cnn2d_recover(dt)
        key_list = ['mixed_y', 'true_y', 'pred_y'] if 'true_y' in dt.keys() else  ['mixed_y', 'pred_y']
        for key in key_list:
            #dt[key] = inverse_stft(dt[key], frame_length=self.win_length, frame_step=self.hop_length, fft_length=self.n_fft, window_fn=self.istft_window)
            # todo: temporary solution with current settings below
            dt[key] = inverse_stft(dt[key], frame_length=self.win_length, frame_step=self.hop_length, fft_length=self.n_fft, window_fn=hann_window) * 2 / 3
        return dt           
   
class ChunkDatav3(Model):

    def __init__(self, chunk_size, target, train, device):
        super(ChunkDatav3, self).__init__()
        self.chunk_size = chunk_size
        self.target = target
        self.train = train
        self.device = device
    
    def call(self, dt):
        with tf.device(self.device):
            for key in ['mixed_mag_comp', 'mixed_mag']:
                time, freq = dt[key].shape
                chunks = time // self.chunk_size + 1
                dt[key] = tf.pad(dt[key],[[0,self.chunk_size - time % self.chunk_size],[0,0]])

            if self.train:
                dt[self.target] = tf.pad(dt[self.target],[[0,self.chunk_size - time % self.chunk_size],[0,0]])

            x_list = []
            y_list = []

            for i in range(chunks):
                x_list.append(dt['mixed_mag_comp'][i * self.chunk_size : (i + 1) * self.chunk_size, : 256])
            dt['x'] = tf.stack(x_list)
            
            if self.train:
                for i in range(chunks):
                    y_list.append(dt[self.target][i * self.chunk_size : (i + 1) * self.chunk_size, : 256])
                dt['y'] = tf.stack(y_list)
            
            dt['x'] =  tf.transpose(dt['x'], perm=[0, 2, 1])

        return dt   
    
def load_model(model, device, action='train', **kargs):
    
    if action == 'train':      
        print('training from scratch')
        epoch = 1
        return epoch, model, optimizer
    
    #elif action == 'retrain':
        # not implemented yet, todo
    
    elif action == 'predict':
        print(f"load model from {kargs['pretrained_model_path']}")
        model.model.load_weights(kargs['pretrained_model_path'])
        return model
