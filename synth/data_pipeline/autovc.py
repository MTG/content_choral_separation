# Generator functions to generate batches of data.

import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections

from synth.config import config
from synth.utils import utils




def data_gen_vc(mode = 'Train', sec_mode = 0):

    datasets = "".join("_"+x.lower() for x in config.datasets)

    with h5py.File(config.stat_file, mode='r') as stat_file:
        max_feat = stat_file["feats_maximus"][()] + 0.001
        min_feat = stat_file["feats_minimus"][()] - 0.001

    voc_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.split('_')[0].upper() in [x for x in config.datasets if x != "DAMP"]]
    if "DAMP" in config.datasets:
        damp_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.split('_')[1] in config.damp_singers]
        voc_list = voc_list+damp_list

    # voc_list = [x for x in voc_list if x not in ['csd_alto1_NinoDios_14.hdf5', 'jvs_jvs023_raw_song_unique_11.hdf5', 'jvs_jvs024_raw_song_unique_2.hdf5', 'csd_soprano3_NinoDios_18.hdf5', 'csd_tenor1_ElRossinyol_13.hdf5', 'csd_soprano3_NinoDios_5.hdf5', 'csd_tenor3_NinoDios_8.hdf5', 'csd_tenor2_NinoDios_13.hdf5', 'jvs_jvs047_raw_song_unique_4.hdf5', 'jvs_jvs098_raw_song_unique_1.hdf5', 'jvs_jvs023_raw_song_unique_9.hdf5', 'jvs_jvs023_raw_song_unique_14.hdf5', 'csd_soprano2_NinoDios_13.hdf5', 'csd_tenor4_LocusIste_12.hdf5', 'csd_bass4_NinoDios_5.hdf5', 'jvs_jvs014_raw_song_unique_15.hdf5', 'csd_soprano2_NinoDios_2.hdf5', 'csd_bass4_NinoDios_12.hdf5', 'jvs_jvs041_raw_song_unique_14.hdf5', 'csd_alto3_LocusIste_25.hdf5', 'jvs_jvs023_raw_song_unique_16.hdf5', 'jvs_jvs092_raw_song_unique_12.hdf5', 'jvs_jvs074_raw_song_unique_6.hdf5', 'jvs_jvs017_raw_song_unique_2.hdf5']]

    train_list = [x for x in voc_list if not x.split('_')[2]=='04'] + voc_list[:int(len(voc_list)*0.9)]
    val_list = [x for x in voc_list if x.split('_')[2]=='04']+ voc_list[int(len(voc_list)*0.9):]


    max_files_to_process = int(config.batch_size/config.autovc_samples_per_file)

    if mode == "Train":
        num_batches = config.autovc_batches_per_epoch_train
        file_list = train_list

    else: 
        num_batches = config.autovc_batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        feats_targs = []

        targets_speakers = []

        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            with h5py.File(os.path.join(config.feats_dir,voc_to_open), "r") as hdf5_file:
                mel = hdf5_file['feats'][()]

            f0 = mel[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            mel[:,-2] = f0

            if np.isnan(mel).any():
                mel = np.nan_to_num(mel)
                print("nan found")
                import pdb;pdb.set_trace()

            speaker_name = voc_to_open.split('_')[1]
            speaker_index = config.singers.index(speaker_name)

            mel = (mel - min_feat)/(max_feat-min_feat)

            assert mel.max()<=1.0 and mel.min()>=0.0, "Error in file {}, max: {}, min: {}".format(voc_to_open, mel.max(), mel.min())


            for j in range(config.autovc_samples_per_file):
                voc_idx = np.random.randint(0,len(mel)-config.max_phr_len)
                feats_targs.append(mel[voc_idx:voc_idx+config.max_phr_len])

                targets_speakers.append(speaker_index)



        feats_targs = np.array(feats_targs)
        

        yield feats_targs, np.array(targets_speakers)



def get_stats():
    """
    Get the maximum and minimum feat values for the datasets to use. 
    """
    datasets = "".join("_"+x.lower() for x in config.datasets)

    voc_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.split('_')[0].upper() in [x for x in config.datasets if x != "DAMP"]]
    if "DAMP" in config.datasets:
        damp_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.split('_')[1] in config.damp_singers]
        voc_list = voc_list+damp_list



    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    count = 0

    too_small = []
 

    for count, voc_to_open in enumerate(voc_list):

        with h5py.File(os.path.join(config.feats_dir,voc_to_open), "r") as voc_file:

            feats = voc_file["feats"][()]

        if len(feats) <= config.max_phr_len or np.isnan(feats).any():
            too_small.append(voc_to_open)
            # import pdb;pdb.set_trace()
            os.remove(os.path.join(config.feats_dir,voc_to_open))
            print("Deleted file {}".format(voc_to_open))
        else:
            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            feats[:,-2] = f0

            maxi_voc_feat = np.array(feats).max(axis=0)

            for i in range(len(maxi_voc_feat)):
                if maxi_voc_feat[i]>max_feat[i]:
                    max_feat[i] = maxi_voc_feat[i]

            mini_voc_feat = np.array(feats).min(axis=0)

            for i in range(len(mini_voc_feat)):
                if mini_voc_feat[i]<min_feat[i]:
                    min_feat[i] = mini_voc_feat[i] 
            count+=1

        utils.progress(count, len(voc_list), "Processed")  

    with h5py.File(config.stat_file, mode='w') as hdf5_file:

        hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
        hdf5_file.create_dataset("feats_minimus", [66], np.float32)
        hdf5_file["feats_maximus"][:] = max_feat
        hdf5_file["feats_minimus"][:] = min_feat
    config.change_variable("stat_prep", "prep", "True")






def main():
    # gen_train_val()
    get_stats()
    # gen = data_gen_vc('Train')
    # while True :
    #     start_time = time.time()
    #     feats_targs, targets_singers = next(gen)
    #     print(time.time()-start_time)


if __name__ == '__main__':
    main()
