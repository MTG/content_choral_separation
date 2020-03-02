# Tests for mutations pertaining to properties.
import os
import librosa

from synth.training import train_autovc, train_SDN, train_autovc_notes
from synth.synthesis import synth_autovc, synth_SDN, synth_autovc_notes
from synth.config import config
from synth.data_prep import prepare_nus
from synth.models import SDN, autovc_emb
# from synth.data_pipeline import autovc_emb

model = autovc_emb.AutoVC()
# model.train()
model.test_file_hdf5("nus_VKOW_11_22.hdf5", 1)

# jaja, jojo = next(autovc_emb.data_gen_vc())
# import pdb;pdb.set_trace()
# prepare_nus.process_dataset()
# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
# train_SDN.train()
# synth_autovc_notes.synthesize_hdf5("nus_ADIZ_09_13.hdf5", 0)
# synth_SDN.synthesize_wav("../datasets/damp_emir_ex/474572_111629-557387728_1622049303-GB-F-1350349314.m4a", 4)

# synth_SDN.synthesize_wav("./DG_take1_3altos.wav", 4)
# synth_SDN.synthesize_wav("../datasets/EsmucChoralSet/2_SeeleChristi/IsolatedSections/SC3_tenors1/SC3_tenors1_AB.wav", 1)
# features = [val for sublist in [x['feats'] for x in [output_dir[y] for y in output_dir.keys()]] for val in sublist] 
# feats = [val for sublist in features for val in sublist] 
# audio, fs = librosa.core.load("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", sr=config.fs)
# model = SDN.SDN()
# model.test_file_wav_f0('../datasets/2020_unison/data/DG_take1_A1.wav','../datasets/2020_unison/data/DG_take1_A1.f0', 1)
# feats = model.extract_feature_wav(audio)
