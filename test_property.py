# Tests for mutations pertaining to properties.
import os
import librosa

from synth.training import train_autovc, train_SDN, train_autovc_notes,train_autovc_notes_emb, train_SDN_notes, train_phone
from synth.synthesis import synth_autovc, synth_SDN, synth_autovc_notes, synth_autovc_notes_emb, synth_SDN_notes, synth_phone
from synth.config import config
from synth.data_prep import prepare_nus, prepare_csd, prepare_damp, prepare_jvs, prepare_yam
from synth.models import SDN, autovc_emb, SIN, autovc, autovc_f0
# from synth.data_pipeline import SDN
from synth.data_pipeline import phone
from synth.utils import sig_process

# gene = autovc_notes_emb.data_gen_vc()
# a,b, c = next(gene)
# import pdb;pdb.set_trace()
# phone.get_stats()
# train_phone.train()
# synth_phone.synthesize_wav('./test_nus.wav')
# model = autovc_emb.AutoVC()
# model.test_file_wav('yam_Scott_Scott_004_8.hdf5', 45)
model = SIN.SIN()
# model.train()
# model.test_file_wav('yam_Scott_Scott_004_10.hdf5', 4)
model.test_file_wav('../sep_content/Unison_analysis_examples/uni_alto_loc.wav')

# jaja, jojo = next(autovc_emb.data_gen_vc())
# import pdb;pdb.set_trace()
# prepare_yam.process_dataset()
# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
# train_autovc.train()
# train_SDN_notes.train()
# train_autovc_notes.train()
# synth_autovc.synthesize_hdf5('yam_Scott_Scott_004_8.hdf5', 45)
# synth_autovc.synthesize_hdf5('nus_JLEE_11_0.hdf5', 1)
# synth_autovc.synthesize_hdf5('nus_MPOL_05_30.hdf5', 2)
# model = autovc.AutoVC()
# model.solo_unison_file_hdf5("csd_alto4_ElRossinyol_0.hdf5", 0.5, 4, 5)
# model.solo_unison_file_hdf5("csd_bass1_NinoDios_8.hdf5", 0.5, 4, 5)
# model.solo_unison_file_hdf5("csd_soprano4_NinoDios_7.hdf5", 0.5, 4, 5)
# model.solo_unison_file_hdf5("csd_tenor1_ElRossinyol_1.hdf5", 0.5, 4, 5)

# synth_SDN_notes.synthesize_wav("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", 1)

# synth_SDN.synthesize_wav("./DG_take1_3altos.wav", 4)
# synth_SDN.synthesize_wav("../datasets/EsmucChoralSet/2_SeeleChristi/IsolatedSections/SC3_tenors1/SC3_tenors1_AB.wav", 1)
# features = [val for sublist in [x['feats'] for x in [output_dir[y] for y in output_dir.keys()]] for val in sublist] 
# feats = [val for sublist in features for val in sublist] 
# audio, fs = librosa.core.load("../datasets/mer_sep_acp/AllAlongTheWatchtower_1.wav", sr=config.fs)
# feats = sig_process.get_world_feats(audio)
# import pdb;pdb.set_trace()
# model = SIN.SIN()
# # model.save_f0_wav('../datasets/unisons/CSD_ER_3sopranos.wav')
# # model.test_file_wav_f0('../datasets/unisons/CSD_ER_3sopranos.wav', "../datasets/unisons/CSD_ER_3sopranos_CREPE.f0")
# feats = model.extract_feature_wav(audio)
# import pdb;pdb.set_trace()
