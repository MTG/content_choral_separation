# Code to prepare the DAMP intonation dataset, go through all the files, process each file and save the data

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
    full_dir = os.path.join(wav_dir, 'vocal_tracks')
    songs = [x for x in os.listdir(full_dir) if x.endswith('m4a') and not x.startswith('.')]


    df = pd.read_csv(os.path.join(wav_dir, "intonation.csv"))
    singers = []

    for count, lf in enumerate(songs):
        song_name = lf.split('.')[0]
        singer_name = df.query('performance_id == "{}"'.format(songs[0].split('.')[0]))[' account_id'].values[0].strip()
        if singer_name in config.damp_singers:
            singers.append(singer_name)
            song_name = song_name.replace('_', '-')
            utils.progress(count, len(songs), "folder processed")
            audio, fs = audio_process.load_audio(os.path.join(full_dir, lf))
            try:
                segments, timestamps, feat, note, stft = audio_process.process_audio(audio)

                for j, (fea, nots, stf)  in enumerate(zip(feat, note, stft)):
                    singer_dict = {}
                    feat[j], note[j], stft[j] = utils.match_time([fea, nots, stf])

                    singer_dict['feats'] = feat[j]
                    singer_dict['notes'] = note[j]
                    singer_dict['stfts'] = stft[j]
                    write_data.write_data(singer_dict, "damp_{}_{}_{}.hdf5".format(singer_name, song_name, j))
            except:
                print("Error in file {}".format(song_name))

    # with open('./singers_DAMP.txt', 'w') as sing_file:
    #     sing_file.write(str(singers))





def process_dataset():
    """
    Go through the entire DAMP intonation dataset and process the data.
    """
    if not config.damp_prep:
        wav_dir = config.raw_dirs['damp_intonation']
        walk_directory(wav_dir)
        config.change_variable("data_prep", "DAMP", "True")
        config.change_variable("stat_prep", "prep", "False")
    else:
        print("DAMP dataset already prepared")


