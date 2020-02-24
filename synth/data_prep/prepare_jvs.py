# Code to prepare the NUS dataset, go through all the files, process each file and save the data

import os,re
import collections
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py

from synth.utils import sig_process, segment, vamp_notes, audio_process, utils, midi_process, write_data
from synth.config import config

def walk_directory(singer_name: str, mode: str):
    """
    Go through a singer directory, processing all files in the directory.
    Arguments:
        singer_name: The name of the singer.
    """
    print("Processing data for JVS singer {}".format(singer_name))

    wav_dir = config.raw_dirs['jvs_music']
    full_dir = os.path.join(wav_dir, singer_name, mode, 'wav')
    sing_wav_files = [x for x in os.listdir(full_dir) if x.endswith('.wav') and x.startswith('raw')]

    for count, lf in enumerate(sing_wav_files):
        utils.progress(count, len(sing_wav_files), "folder processed")
        audio, fs = audio_process.load_audio(os.path.join(full_dir, lf))

        segments, timestamps, feat, note, stft = audio_process.process_audio(audio)

        for j, (fea, nots, stf)  in enumerate(zip(feat, note, stft)):
            singer_dict = {}
            feat[j], note[j], stft[j] = utils.match_time([fea, nots, stf])

            singer_dict['feats'] = feat[j]
            singer_dict['notes'] = note[j]
            singer_dict['stfts'] = stft[j]
            write_data.write_data(singer_dict, "jvs_{}_{}_{}_{}.hdf5".format(singer_name, lf[:-4],mode, j))




def process_dataset():
    """
    Go through the entire NUS dataset and process the data.
    """
    if not config.jvs_prep:
        wav_dir = config.raw_dirs['jvs_music']
        singers = next(os.walk(wav_dir))[1]
        for count, singer in enumerate(singers, 1):
            print("Processing singer {} of {}".format(count, len(singers)))
            walk_directory(singer, 'song_common')
            walk_directory(singer, 'song_unique')
        config.change_variable("data_prep", "JVS", "True")
        config.change_variable("stat_prep", "prep", "False")
    else:
        print("JVS dataset already prepared")





