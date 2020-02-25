# Tests for mutations pertaining to properties.
import os
import librosa

from synth.training import train_autovc, train_SDN
from synth.synthesis import synth_autovc, synth_SDN
from synth.config import config
from synth.data_prep import prepare_csd
from synth.models import SDN
# prepare_csd.process_dataset()
# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
# train_SDN.train()
# synth_autovc.synthesize_hdf5("nus_JTAN_15_19.hdf5", 3)
# synth_SDN.synthesize_wav("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", 1)
# features = [val for sublist in [x['feats'] for x in [output_dir[y] for y in output_dir.keys()]] for val in sublist] 
# feats = [val for sublist in features for val in sublist] 
audio, fs = librosa.core.load("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", sr=config.fs)
model = SDN.SDN()
feats = model.extract_feature_wav(audio)