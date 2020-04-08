# Tests for mutations pertaining to properties.
import os
import librosa

from synth.training import train_autovc, train_SDN, train_autovc_notes,train_autovc_notes_emb, train_SDN_notes
from synth.synthesis import synth_autovc, synth_SDN, synth_autovc_notes, synth_autovc_notes_emb, synth_SDN_notes
from synth.config import config
from synth.data_prep import prepare_musdb
from synth.models import SDN, autovc_emb, SIN
# from synth.data_pipeline import SDN
from synth.data_pipeline import autovc_notes_emb

# gene = autovc_notes_emb.data_gen_vc()
# a,b, c = next(gene)
# import pdb;pdb.set_trace()


# model = autovc_emb.AutoVC()
# model = SIN.SIN()
# model.train()
# model.test_file_hdf5("nus_ADIZ_09_13.hdf5", 1)

# jaja, jojo = next(autovc_emb.data_gen_vc())
# import pdb;pdb.set_trace()
# prepare_musdb.process_dataset()
# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
# train_autovc_notes_emb.train()
# train_SDN_notes.train()
# synth_autovc_notes.synthesize_hdf5('damp_469436485_469439275-1774003938_2.hdf5', 25)
# synth_autovc.synthesize_hdf5('nus_KENN_04_15.hdf5', 5)

# synth_SDN_notes.synthesize_wav("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", 1)

# synth_SDN.synthesize_wav("./DG_take1_3altos.wav", 4)
# synth_SDN.synthesize_wav("../datasets/EsmucChoralSet/2_SeeleChristi/IsolatedSections/SC3_tenors1/SC3_tenors1_AB.wav", 1)
# features = [val for sublist in [x['feats'] for x in [output_dir[y] for y in output_dir.keys()]] for val in sublist] 
# feats = [val for sublist in features for val in sublist] 
audio, fs = librosa.core.load("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", sr=config.fs)
model = SIN.SIN()
# model.save_f0_wav('../datasets/unisons/CSD_ER_3sopranos.wav')
# model.test_file_wav_f0('../datasets/unisons/CSD_ER_3sopranos.wav', "../datasets/unisons/CSD_ER_3sopranos_CREPE.f0")
feats = model.extract_feature_wav(audio)
import pdb;pdb.set_trace()
