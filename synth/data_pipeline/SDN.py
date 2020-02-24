# Generator functions to generate batches of data.

import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections

from synth.config import config
from synth.utils import utils




def data_gen_SDN(mode = 'Train', sec_mode = 0):

    with h5py.File(config.stat_file, mode='r') as stat_file:
        max_feat = stat_file["feats_maximus"][()] + 0.001
        min_feat = stat_file["feats_minimus"][()] - 0.001

    voc_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.split('_')[0].upper() in config.datasets]


    train_list = [x for x in voc_list if not x.split('_')[2]=='04']
    val_list = [x for x in voc_list if x.split('_')[2]=='04']


    max_files_to_process = int(config.batch_size/config.autovc_samples_per_file)

    if mode == "Train":
        num_batches = config.autovc_batches_per_epoch_train
        file_list = train_list

    else: 
        num_batches = config.autovc_batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        feats_targs = []
        stfts_targs = []
        targets_speakers = []

        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            with h5py.File(os.path.join(config.feats_dir,voc_to_open), "r") as hdf5_file:
                mel = hdf5_file['feats'][()]
                stfts = hdf5_file['stfts'][()]

            f0 = mel[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            mel[:,-2] = f0


            speaker_name = voc_to_open.split('_')[1]
            speaker_index = config.singers.index(speaker_name)

            mel = (mel - min_feat)/(max_feat-min_feat)

            stfts = np.clip(stfts, 0.0, 1.0)

            assert mel.max()<=1.0 and mel.min()>=0.0, "Error in file {}, max: {}, min: {}".format(voc_to_open, mel.max(), mel.min())


            for j in range(config.autovc_samples_per_file):
                voc_idx = np.random.randint(0,len(mel)-config.max_phr_len)
                feats_targs.append(mel[voc_idx:voc_idx+config.max_phr_len])
                noise = np.random.rand(config.max_phr_len,stfts.shape[-1])*np.clip(np.random.rand(1),0.0,config.noise_threshold)
                stfts_targs.append(stfts[voc_idx:voc_idx+config.max_phr_len] + noise)
                targets_speakers.append(speaker_index)



        feats_targs = np.array(feats_targs)
        stfts_targs = np.array(stfts_targs)
        

        yield feats_targs, stfts_targs, np.array(targets_speakers)