# Tests for mutations pertaining to properties.
import os
import librosa

from synth.training import train_autovc, train_SDN, train_autovc_notes
from synth.synthesis import synth_autovc, synth_SDN, synth_autovc_notes
from synth.config import config
from synth.data_prep import prepare_nus
from synth.models import SDN
# prepare_nus.process_dataset()
# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
train_autovc.train()
# synth_autovc_notes.synthesize_hdf5("nus_ADIZ_01_0.hdf5", 4)
# synth_SDN.synthesize_wav("../sep_content/nino_alto_1_3singers.wav", 4)
# synth_SDN.synthesize_wav("../datasets/EsmucChoralSet/2_SeeleChristi/IsolatedSections/SC3_tenors1/SC3_tenors1_AB.wav", 1)
# features = [val for sublist in [x['feats'] for x in [output_dir[y] for y in output_dir.keys()]] for val in sublist] 
# feats = [val for sublist in features for val in sublist] 
# audio, fs = librosa.core.load("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", sr=config.fs)
# model = SDN.SDN()
# feats = model.extract_feature_wav(audio)