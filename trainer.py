from tqdm import tqdm
import os
from utils import evaluation
import numpy as np
import tensorflow as tf

class Trainer():

    def __init__(self, model, train_loader, valid_loader, device):
        self.model = model
        self.train_set = train_loader
        self.valid_set = valid_loader
        self.device = device
        self.count = 0
        
    def train(self, epoch_to_start_from, total_epochs, *args, **kwargs):
        """
        - epoch_to_start_from: starting epoch
        - total_epochs: run this many epochs
        """
        best_snr = 0
        best_pesq = 0
        best_stoi = 0

        for epoch in range(epoch_to_start_from, epoch_to_start_from + total_epochs + 1):

            train_loss = self._train(epoch) 
            loss, pesq, stoi, segsnr = self._valid(epoch)
            if not os.path.exists(kwargs['model_path']):
                os.mkdir(kwargs['model_path'])

            self.model.model.save(os.path.join(kwargs['model_path'], f'epoch{epoch}'))

            if best_snr < segsnr or best_pesq < pesq or best_stoi < stoi:
                # save tensorflow model
                self.model.model.save(os.path.join(kwargs['model_path'], f'best_model_epoch{epoch}'))
                best_snr = segsnr if best_snr < segsnr else best_snr
                best_stoi = stoi if best_stoi < stoi else best_stoi
                best_pesq = pesq if best_pesq < pesq else best_pesq
    
    def _train(self, epoch):

        total_loss = 0
        total_accuracy = 0
        self.train_set.on_epoch_begin()
        train_bar = tqdm(self.train_set)     
        #self.model.train()

        for batch in train_bar:
            train_bar.set_description(f'Epoch {epoch} train')
            
            batch = self.model.stft(batch)
            batch = self.model.chunk(batch)
            with tf.device(self.device):
                x = tf.expand_dims(batch['x'],-1)
                y = batch['y']
                test_output = self.model.model.test_on_batch(x, y)
                history = self.model.model.train_on_batch(x, y)
                total_loss += test_output[0]
                total_accuracy += history[1]
            train_bar.set_postfix(loss=test_output[0], acc=history[1])     
        lens = len(self.train_set)
        average_loss = round(total_loss / lens, 4)
        average_accuracy = round(total_accuracy / lens, 4)
        print('\tLoss: ', average_loss)
        with open("results.txt", "a") as myfile:
            myfile.write(f'Training Epoch:{epoch}, Loss: {average_loss}, Accuracy: {average_accuracy}\n')
        self.train_set.on_epoch_end()
        return average_loss

    def _valid(self, epoch):

        total_loss = 0
        total_pesq, total_stoi, total_segsnr = 0, 0, 0
        valid_bar = tqdm(self.valid_set)
        bce = tf.keras.losses.BinaryCrossentropy()

        for batch in valid_bar:

            valid_bar.set_description(f'Epoch {epoch} valid')

            with tf.device(self.device):
                batch = self.model.stft(batch)
                batch = self.model.chunk(batch)
                batch['pred_mask'] = self.model.model(tf.expand_dims(batch['x'],-1))
                batch = self.model.istft(batch)
                local_loss = bce(batch['y'], batch['pred_mask']).numpy()
                total_loss += local_loss
                pesq, stoi, segsnr = evaluation(np.squeeze(batch['true_y'].numpy()), batch['pred_y'].numpy())
                total_pesq += pesq
                total_stoi += stoi
                total_segsnr += segsnr
                valid_bar.set_postfix(loss=round(local_loss, 2), pesq=pesq, stoi=stoi, segSNR=segsnr)

        lens = len(self.valid_set)
        average_loss = round(total_loss / lens, 4)
        average_pesq = round(total_pesq / lens, 2)
        average_stoi = round(total_stoi / lens, 2)
        average_segsnr = round(total_segsnr / lens, 2)
        print('\tLoss: ', average_loss, 'pesq: ', average_pesq, 'stoi: ', average_stoi, 'sngSNR: ', average_segsnr)
        with open("results.txt", "a") as myfile:
            myfile.write(f'Loss:{average_loss}, pesq: {average_pesq}, stoi: {average_stoi}, sngSNR: {average_segsnr} \n')
        return average_loss, average_pesq, average_stoi, average_segsnr
