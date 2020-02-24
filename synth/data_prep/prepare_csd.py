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

def walk_directory(song_name: str):
    """
    Go through a song directory, processing all files in the directory.
    Arguments:
        singer_name: The name of the song.
    """
    print("Processing data for CSD song {}".format(song_name))

    wav_dir = config.raw_dirs['choralsingingdataset']
    full_dir = os.path.join(wav_dir, song_name, 'IndividualVoices')
    sing_wav_files = [x for x in os.listdir(full_dir) if x.endswith('.wav') and not x.startswith('.') and not x.endswith('-24b.wav')]

    for count, lf in enumerate(sing_wav_files):
        singer_name = lf.split('_')[1]+lf.split('_')[2].replace('.wav','')
        utils.progress(count, len(sing_wav_files), "folder processed")
        audio, fs = audio_process.load_audio(os.path.join(full_dir, lf))

        segments, timestamps, feat, note, stft = audio_process.process_audio(audio)

        for j, (fea, nots, stf)  in enumerate(zip(feat, note, stft)):
            singer_dict = {}
            feat[j], note[j], stft[j] = utils.match_time([fea, nots, stf])

            singer_dict['feats'] = feat[j]
            singer_dict['notes'] = note[j]
            singer_dict['stfts'] = stft[j]
            write_data.write_data(singer_dict, "csd_{}_{}_{}.hdf5".format(singer_name, song_name, j))




def process_dataset():
    """
    Go through the entire NUS dataset and process the data.
    """
    if not config.csd_prep:
        wav_dir = config.raw_dirs['choralsingingdataset']
        songs = ['ElRossinyol', 'LocusIste', 'NinoDios']
        for count, song in enumerate(songs, 1):
            print("Processing song {} of {}".format(count, len(songs)))
            walk_directory(song)
        config.change_variable("data_prep", "CSD", "True")
        config.change_variable("stat_prep", "prep", "False")
    else:
        print("CSD dataset already prepared")


