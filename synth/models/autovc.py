# Script for training and synthesizing from the AutoVC model
import tensorflow as tf
import numpy as np
import librosa
import pyworld as pw
import sys
import os,re
import soundfile as sf
import matplotlib.pyplot as plt
import time
import h5py

from synth.data_pipeline import autovc as data_pipeline
from synth.config import config
from . import model
from synth.modules import autovc as modules
from synth.utils import utils, sig_process



            

class AutoVC(model.Model):

    def __init__(self):
        self.check_prep()
        self.get_placeholders()
        self.model()
        self.sess = tf.Session()
        summary_dict = self.loss_function()
        self.get_optimizers()
        self.get_summary(self.sess, config.autovc_log_dir, summary_dict)
        self.load_model(self.sess, config.autovc_log_dir)

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.optimizer.minimize(self.final_loss, global_step = self.global_step)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        self.recon_loss = tf.reduce_sum(tf.square(self.input_placeholder - self.output) ) 


        self.content_loss = tf.reduce_sum(tf.abs(self.content_embedding_1 - self.content_embedding_2))

        self.recon_loss_0 = tf.reduce_sum(tf.square(self.input_placeholder - self.output_1))

        self.final_loss = self.recon_loss + config.mu * self.recon_loss_0 + config.lamda * self.content_loss

        summary_dict = {"recon_loss" : self.recon_loss, "content_loss": self.content_loss, "recon_loss_0": self.recon_loss_0, "final_loss": self.final_loss}

        return summary_dict



    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.num_features),
                                           name='input_placeholder')       


        self.speaker_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels = tf.one_hot(indices=tf.cast(self.speaker_labels, tf.int32), depth = config.num_singers)

        self.speaker_labels_1 = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels_1 = tf.one_hot(indices=tf.cast(self.speaker_labels_1, tf.int32), depth = config.num_singers)

        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
 

        start_epoch = int(self.sess.run(tf.train.get_global_step()) / (config.autovc_batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.autovc_num_epochs):

            data_generator = data_pipeline.data_gen_vc()
            val_generator = data_pipeline.data_gen_vc(mode = 'Val')
            

            epoch_final_loss = 0
            epoch_recon_loss = 0
            epoch_recon_0_loss = 0
            epoch_content_loss = 0

            val_final_loss = 0
            val_recon_loss = 0
            val_recon_0_loss = 0
            val_content_loss = 0

            batch_num = 0

            start_time = time.time()

            with tf.variable_scope('Training'):
                for feats_targs, targets_speakers in data_generator:


                    final_loss, recon_loss, recon_loss_0, content_loss,  summary_str = self.train_model(feats_targs, targets_speakers, self.sess)

                    epoch_final_loss+=final_loss
                    epoch_recon_loss+=recon_loss
                    epoch_recon_0_loss+=recon_loss_0
                    epoch_content_loss+=content_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.autovc_batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_recon_loss = epoch_recon_loss/batch_num
                epoch_recon_0_loss = epoch_recon_0_loss/batch_num
                epoch_content_loss = epoch_content_loss/batch_num

                print_dict = {"Final Loss": epoch_final_loss}

                print_dict["Recon Loss"] =  epoch_recon_loss
                print_dict["Recon Loss_0 "] =  epoch_recon_0_loss
                print_dict["Content Loss"] =  epoch_content_loss


            batch_num = 0
            with tf.variable_scope('Validation'):
                for feats_targs, targets_speakers in val_generator:


                    final_loss, recon_loss, recon_loss_0, content_loss,  summary_str = self.validate_model(feats_targs, targets_speakers, self.sess)

                    val_final_loss+=final_loss
                    val_recon_loss+=recon_loss
                    val_recon_0_loss+=recon_loss_0
                    val_content_loss+=content_loss

                    self.val_summary_writer.add_summary(summary_str, epoch)
                    self.val_summary_writer.flush()

                    utils.progress(batch_num,config.autovc_batches_per_epoch_val, suffix = 'validation done')

                    batch_num+=1

                val_final_loss = val_final_loss/batch_num
                val_recon_loss = val_recon_loss/batch_num
                val_recon_0_loss = val_recon_0_loss/batch_num
                val_content_loss = val_content_loss/batch_num

                print_dict["Val Final Loss"] = val_final_loss

                print_dict["Val Recon Loss"] =  val_recon_loss
                print_dict["Val Recon Loss_0 "] =  val_recon_0_loss
                print_dict["Val Content Loss"] =  val_content_loss



            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.autovc_num_epochs:
                self.save_model(self.sess, epoch+1, config.autovc_log_dir)

    def train_model(self,feats_targs, targets_speakers, sess):
        """
        Function to train the model for each epoch
        """


        feed_dict = {self.input_placeholder: feats_targs[:,:,:-2], self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers,  self.is_train: True}
            
        _, final_loss, recon_loss, recon_loss_0, content_loss = sess.run([self.final_train_function,self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, summary_str
 

    def validate_model(self,feats_targs, targets_speakers, sess):
        """
        Function to validate the model for each epoch
        """

        feed_dict = {self.input_placeholder: feats_targs[:,:,:-2], self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers,  self.is_train: False}

            
        final_loss, recon_loss, recon_loss_0, content_loss = sess.run([self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):


        with h5py.File(os.path.join(config.feats_dir,file_name), "r") as hdf5_file:
            mel = hdf5_file['feats'][()]

        f0 = mel[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        mel[:,-2] = f0

        return mel

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        voc_stft = abs(utils.stft(vocals))

        feats = utils.stft_to_feats(vocals,fs)

        voc_stft = np.clip(voc_stft, 0.0, 1.0)

        return voc_stft, feats



    def test_file_wav(self, file_name, speaker_index):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """

        stft, mel = self.read_wav_file(file_name)

        out_mel = self.process_file(stft, speaker_index, self.sess)


        self.plot_features(mel, out_mel)



    def test_file_hdf5(self, file_name, speaker_index_2):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """

        mel = self.read_hdf5_file(file_name)

        speaker_name = file_name.split('_')[1]
        speaker_index = config.singers.index(speaker_name)


        out_mel = self.process_file(mel, speaker_index, speaker_index_2, self.sess)

        plot_dict = {"Spec Envelope": {"gt": mel[:,:-6], "op": out_mel[:,:-4]}, "Aperiodic":{"gt": mel[:,-6:-2], "op": out_mel[:,-4:]}}


        self.plot_features(plot_dict)



        synth = utils.query_yes_no("Synthesize output? ")

        if synth:
            gen_change = utils.query_yes_no("Change in gender? ")
            if gen_change:
                female_male = utils.query_yes_no("Female to male?")
                if female_male:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]-12, mel[:out_mel.shape[0],-1:]), axis = -1)
                else:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]+12, mel[:out_mel.shape[0],-1:]), axis = -1)
            else:
                out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1], mel[:out_mel.shape[0],-1:]), axis = -1)

            audio_out = sig_process.feats_to_audio(out_featss) 

            sf.write(os.path.join(config.output_dir,'./{}_{}_autovc.wav'.format(file_name[:-5], config.singers[speaker_index_2])), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize ground truth with vocoder? ")

        if synth_ori:
            audio = sig_process.feats_to_audio(mel) 
            sf.write(os.path.join(config.output_dir,'./{}_{}_ori.wav'.format(file_name[:-5], config.singers[speaker_index])), audio, config.fs)





    def process_file(self, mel, speaker_index, speaker_index_2, sess):


        datasets = "".join("_"+x.lower() for x in config.datasets)

        with h5py.File(config.stat_file, mode='r') as stat_file:
            max_feat = stat_file["feats_maximus"][()] + 0.001
            min_feat = stat_file["feats_minimus"][()] - 0.001



        mel = (mel - min_feat)/(max_feat-min_feat)

        in_batches_mel, nchunks_in = utils.generate_overlapadd(mel)

        out_batches_mel = []

        for in_batch_mel in in_batches_mel :
            speaker = np.repeat(speaker_index, config.batch_size)
            speaker_2 = np.repeat(speaker_index_2, config.batch_size)
            feed_dict = {self.input_placeholder: in_batch_mel[:,:,:-2], self.speaker_labels:speaker,self.speaker_labels_1:speaker_2, self.is_train: False}
            mel = sess.run(self.output, feed_dict=feed_dict)

            out_batches_mel.append(mel)
        out_batches_mel = np.array(out_batches_mel)

        out_batches_mel = utils.overlapadd(out_batches_mel,nchunks_in)


        out_batches_mel = out_batches_mel*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        return out_batches_mel



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """


        with tf.variable_scope('encoder') as scope:
            self.content_embedding_1 = modules.content_encoder(self.input_placeholder, self.speaker_onehot_labels, self.is_train)


        with tf.variable_scope('decoder') as scope: 
            self.output_1 = modules.decoder(self.content_embedding_1, self.speaker_onehot_labels_1, self.is_train)

        with tf.variable_scope('post_net') as scope: 
            self.residual = modules.post_net(self.output_1, self.is_train)
            self.output = self.output_1 + self.residual
        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_2 = modules.content_encoder(self.output, self.speaker_onehot_labels, self.is_train)