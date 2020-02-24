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

def walk_directory(singer_name: str, mode: str='sing'):
    """
    Go through a singer directory, processing all files in the directory.
    Arguments:
        singer_name: The name of the singer.
        mode: either sing or read.
    """
    print("Processing data for NUS singer {}".format(singer_name))
    print("Processing the {} directory".format(mode))

    wav_dir = config.raw_dirs['nus']
    full_dir = os.path.join(wav_dir, singer_name, mode)
    sing_wav_files = [x for x in os.listdir(full_dir) if x.endswith('.wav') and not x.startswith('.')]

    for count, lf in enumerate(sing_wav_files):
        utils.progress(count, len(sing_wav_files), "folder processed")
        audio, fs = audio_process.load_audio(os.path.join(full_dir, lf))

        segments, timestamps, feat, note, stft = audio_process.process_audio(audio)

        phonemes = midi_process.open_lab_file(os.path.join(full_dir, lf[:-4]+".txt"))

        phos = np.array(midi_process.pho_segment_allign(phonemes, timestamps))

        for j, (fea, nots, stf, pho)  in enumerate(zip(feat, note, stft, phos)):
            singer_dict = {}
            feat[j], note[j], stft[j], phos[j] = utils.match_time([fea, nots, stf, pho])

            singer_dict['feats'] = feat[j]
            singer_dict['notes'] = note[j]
            singer_dict['phons'] = phos[j]
            singer_dict['stfts'] = stft[j]
            write_data.write_data(singer_dict, "nus_{}_{}_{}.hdf5".format(singer_name, lf[:-4], j))




def process_dataset(mode: str='sing'):
    """
    Go through the entire NUS dataset and process the data.
    """
    if not config.nus_prep:
        wav_dir = config.raw_dirs['nus']
        singers = next(os.walk(wav_dir))[1]

        for count, singer in enumerate(singers, 1):
            print("Processing singer {} of {}".format(count, len(singers)))
            walk_directory(singer)
        config.change_variable("data_prep", "NUS", "True")
        config.change_variable("stat_prep", "prep", "False")
    else:
        print("NUS dataset already prepared")





