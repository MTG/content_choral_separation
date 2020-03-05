# Code to prepare the Bach Chorales dataset, go through all the files, process each file and save the data

import os,re
import collections
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import pandas as pd

from synth.utils import sig_process, segment, vamp_notes, audio_process, utils, midi_process, write_data
from synth.config import config

def walk_directory(wav_dir):
    """
    Go through a song directory, processing all files in the directory.
    Arguments:
        singer_name: The name of the song.
    """
    print("Processing data for the DAMP intonation dataset")
    songs = [x for x in os.listdir(wav_dir) if x.endswith('1ch.wav') and not x.startswith('.')]
    singers = []


    for count, lf in enumerate(songs):
        song_name = lf.split('_')[0]+lf.split('_')[1]
        singer_name = lf.split('_')[3]
        if singer_name not in singers:
            singers.append(singer_name)
        utils.progress(count, len(songs), "folder processed")
        
        try:
            audio, fs = audio_process.load_audio(os.path.join(wav_dir, lf))
            feat, note, stft = audio_process.process_seg(audio)
            singer_dict = {}
            feat, note, stft = utils.match_time([feat, note, stft])
            singer_dict['feats'] = feat
            singer_dict['notes'] = note
            singer_dict['stfts'] = stft
            write_data.write_data(singer_dict, "bach_{}_{}_{}.hdf5".format(singer_name, song_name, lf.split('_')[2].replace("part", "")))
        except:
            print("Error in file {}".format(song_name))

    with open('./singers_BACH.txt', 'w') as sing_file:
        sing_file.write(str(singers))





def process_dataset():
    """
    Go through the entire Bach Chorales dataset and process the data.
    """
    if not config.bach_prep:
        wav_dir = config.raw_dirs['bach']
        walk_directory(wav_dir)
        config.change_variable("data_prep", "BACH", "True")
        config.change_variable("stat_prep", "prep", "False")
    else:
        print("DAMP dataset already prepared")


