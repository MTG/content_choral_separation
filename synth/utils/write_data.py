import os,re
import collections
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py

from synth.utils import sig_process, segment, vamp_notes, audio_process, utils, midi_process
from synth.config import config


def write_data(inputs, file_name, feats_dir=config.feats_dir):
    """
    Function to save data. For now, saving each segment separatly.
    Arguments:
        inputs: A dictionary of input values, should be real valued np arrays.
        file_name: The name of the hdf5 file to be saved. 
    """
    if not os.path.isdir(feats_dir):
        os.mkdir(feats_dir)
    with h5py.File(os.path.join(feats_dir, file_name), mode='w') as hdf5_file:
        for key in inputs.keys():
            inp = inputs[key]
            if len(inp.shape) == 1:
                inp = np.expand_dims(inp, -1)
            hdf5_file.create_dataset(key, inp.shape, inp.dtype)
            hdf5_file[key][:,:] = inp