import configparser
import os
import pandas as pd

if 'CONFIG_PATH' in os.environ.keys():
    config_path = os.environ['CONFIG_PATH']
else:
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.ini')

config = configparser.ConfigParser()

config.read(config_path)

raw_dirs = config["raw_dirs"]
emb_dir = raw_dirs["embs_ge2e_full"]

sig_process = config["signal_processing"]
fs = int(sig_process["fs"])
world_offset = float(sig_process["world_offset"])
hopsize = int(sig_process["hopsize"])
framesize = int(sig_process["framesize"])
silence_ms = int(sig_process["silence_ms"])
hoptime = float(hopsize/fs)
stft_features = int(framesize/2+1)

params = config["params"]
batch_size = int(params["batch_size"])
max_phr_len = int(params["max_phr_len"])
min_phr_len = int(params["min_phr_len"])
num_features = int(params["num_features"])
init_lr = float(params["init_lr"])
max_models_to_keep = int(params["max_models_to_keep"])
print_every = int(params["print_every"])
save_every = int(params["save_every"])

nus_params = config['nus']
phonemas = nus_params['phonemas'].split(', ')
phonemas_nus = nus_params['phonemas2'].split(', ')
nus_singers = nus_params['singers'].split(', ')
nus_genders = nus_params['genders'].split(', ')
nus_genders = {a:b for a,b in zip(nus_singers, nus_genders)}

jvs_params = config['jvs']
jvs_singers = jvs_params['singers'].split(', ')

csd_params = config['csd']
csd_singers = csd_params['singers'].split(', ')

bach_params = config['bach']
bach_singers = bach_params['singers'].split(', ')


    # import pdb;pdb.set_trace()

feature_params = config["feature_params"]
feats_dir = "{}_{}_{}_{}".format(feature_params['feats_dir'], fs, hopsize, framesize)
output_dir = feature_params['output_dir']
backing_dir = "{}_{}_{}_{}".format(feature_params['backing_dir'], fs, hopsize, framesize)


datasets_params = config["datasets"]
datasets = datasets_params["datasets"].split(", ")
dataset_list = "".join("_"+x.lower() for x in datasets)

singers = []
genders = {}
if "NUS" in datasets:
    singers = singers + nus_singers
    genders = {**genders, **nus_genders}

if "JVS" in datasets:
    singers = singers + jvs_singers
if "CSD" in datasets:
    singers = singers + csd_singers
if "DAMP" in datasets:
    with open('/home/pc2752/share/final_research/singers_DAMP.txt') as damp_file:
        damp_singers = damp_file.read().split(', ')
        wav_dir = raw_dirs['damp_intonation']
        df = pd.read_csv(os.path.join(wav_dir, "intonation.csv"))
        damp_singers = [x for x in damp_singers if df[df[" account_id"] == " {}".format(x)][' locale'].values[0].strip().split("_")[0] == "en"][:100]
        damp_genders = [df[df[" account_id"] == " {}".format(x)][' gender'].values[0].strip() for x in damp_singers]
        damp_genders = {a:b for a,b in zip(damp_singers, damp_genders)}

    singers = singers + damp_singers
    genders = {**genders, **damp_genders}
if "BACH" in datasets:
    singers = singers + bach_singers


num_singers = len(singers)


data_prep = config["data_prep"]
nus_prep = data_prep.getboolean("NUS")
csd_prep = data_prep.getboolean("CSD")
jvs_prep = data_prep.getboolean("JVS")
damp_prep = data_prep.getboolean("DAMP")
bach_prep = data_prep.getboolean("BACH")
musdb_prep = data_prep.getboolean("MUSDB")

stat_params = config["stat_prep"]
stat_prep = stat_params.getboolean("prep")
stat_file = '{}{}.hdf5'.format(stat_params["stat_file"], dataset_list)


autovc_params = config["autovc"]
autovc_batches_per_epoch_train = int(autovc_params["batches_per_epoch_train"])
autovc_batches_per_epoch_val =  int(autovc_params["batches_per_epoch_val"])
autovc_num_epochs = int(autovc_params["num_epochs"])
autovc_lstm_size = int(autovc_params["lstm_size"])
autovc_samples_per_file = int(autovc_params["samples_per_file"])
autovc_code_sam = int(autovc_params["code_sam"])
lamda = int(autovc_params['lambda'])
mu = int(autovc_params['mu'])
autovc_log_dir = "{}_{}_{}_{}{}".format(autovc_params['log_dir'], fs, hopsize, framesize, dataset_list)
autovc_notes_log_dir = "{}_notes_{}_{}_{}{}".format(autovc_params['log_dir'], fs, hopsize, framesize, dataset_list)
autovc_log_dir_f0 = "{}_f0_{}_{}_{}{}".format(autovc_params['log_dir'], fs, hopsize, framesize, dataset_list)
autovc_emb_log_dir = "{}_emb_{}_{}_{}{}".format(autovc_params['log_dir'], fs, hopsize, framesize, dataset_list)
autovc_notes_emb_log_dir = "{}_notes_emb_{}_{}_{}{}".format(autovc_params['log_dir'], fs, hopsize, framesize, '_nus')
autovc_mix_emb = autovc_params.getboolean("mix_emb")
autovc_emb_feats = int(autovc_params["emb_features"])

SDN_params = config["SDN"]
SDN_mix = SDN_params.getboolean("mix")
if SDN_mix:
    SDN_log_dir = "{}_{}_{}_{}{}_mix/".format(SDN_params['log_dir'], fs, hopsize, framesize, dataset_list)
    SDN_notes_log_dir = "{}_{}_{}_{}{}_mix_notes/".format(SDN_params['log_dir'], fs, hopsize, framesize, dataset_list)
else:
    SDN_log_dir = "{}_{}_{}_{}{}_nomix/".format(SDN_params['log_dir'], fs, hopsize, framesize, dataset_list)
    SDN_notes_log_dir = "{}_{}_{}_{}{}_nomix_notes/".format(SDN_params['log_dir'], fs, hopsize, framesize, dataset_list)

phone_params = config["PHONE"]
phone_log_dir =  "{}_{}_{}_{}{}/".format(phone_params['log_dir'], fs, hopsize, framesize, dataset_list)


filter_len = int(SDN_params["filter_len"])
encoder_layers = int(SDN_params["encoder_layers"])
filters = int(SDN_params["filters"])
augment_filters_every = int(SDN_params["augment_filters_every"])
noise_threshold = float(SDN_params["noise_threshold"])
back_threshold = float(SDN_params["back_threshold"])
SDN_num_epochs = int(SDN_params["num_epochs"])


SIN_params = config["SIN"]
SIN_mix = SIN_params.getboolean("mix")
if SDN_mix:
    SIN_log_dir = "{}_{}_{}_{}{}_mix/".format(SIN_params['log_dir'], fs, hopsize, framesize, dataset_list)
else:
    SIN_log_dir = "{}_{}_{}_{}{}_nomix/".format(SIN_params['log_dir'], fs, hopsize, framesize, dataset_list)



def change_variable(path, variable, new_value):
    global config
    global config_path
    config[path][variable] = new_value
    with open(config_path, 'w') as configfile:
        config.write(configfile)