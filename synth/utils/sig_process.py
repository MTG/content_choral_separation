# Module for signal processing. 

import os,re
import collections
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import pyworld as pw
from synth.utils.reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp, mgc_to_mfsc, mfsc_to_mgc
from scipy.ndimage import filters
from synth.utils import utils
from synth.config import config

import librosa


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def get_hcqt(audio):
    """
    Get the harmonic CQT for the input audio.
    """

    cqt_list = []
    shapes = []
    for h in config.harmonics:
        
        cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin*float(h), bins_per_octave = config.bins_per_octave)
        cqt_list.append(cqt.T)

    cqt_list = utils.match_time(cqt_list)



    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt


def process_f0(f0, f_bins, n_freqs):
    """
    Get the atb (discrete F0 representation) for an input f0.
    """
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))

    haha = np.digitize(f0, f_bins) - 1

    idx2 = haha < n_freqs

    haha = haha[idx2]

    freqz[range(len(haha)), haha] = 1

    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T

    min_target = np.min(atb[range(len(haha)), haha])

    atb = atb / min_target

    # import pdb;pdb.set_trace()

    atb[atb > 1] = 1

    return atb

def get_world_feats(vocals):
    vocals = np.float64(vocals)
    feats=pw.wav2world(vocals,config.fs,frame_period= config.hoptime*1000)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10*np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    harm+=config.world_offset
    f0 = feats[0]


    # f0 = pitch.extract_f0_sac(vocals, fs, config.hoptime)

    y = f0_to_hertz(f0)
    # import pdb;pdb.set_trace()
    # y = hertz_to_new_base(f0)
    nans, x= utils.nan_helper(y)
    naners=np.isinf(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)

    # if config.comp_mode == 'mfsc':
    harmy=sp_to_mfsc(harm,60,0.45)
    apy=sp_to_mfsc(ap,4,0.45)
    # elif config.comp_mode == 'mgc':
    #     harmy=sp_to_mgc(harm,60,0.45)
    #     apy=sp_to_mgc(ap,4,0.45)

    out_feats=np.concatenate((harmy,apy,y.reshape((-1,2))),axis=1) 

    return out_feats

def f0_to_hertz(f0):

    return 69+12*np.log2(f0/440)

def hertz_to_f0(f0):
    # if f0 == 0:
    #     return 0
    # else:
    f0 = f0-69
    f0 = f0/12
    f0 = 2**f0
    f0 = f0*440
    return f0

def feats_to_audio(in_feats):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    f0[:,0] = hertz_to_f0(f0[:,0])

    f0 = f0[:,0]*(1-f0[:,1])

    wraped_freq = get_warped_freqs(60,config.fs,0.45)
    harm = mfsc_to_sp(harm, wraped_freq,1025,config.fs)
    ap = wbap_to_ap(ap, 1025, config.fs)

    harm = np.ascontiguousarray(10**((harm - config.world_offset)/10))
    ap = np.ascontiguousarray(10**(ap/20))

    y = pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),config.fs,config.hoptime*1000)
    return y


def get_feats(audio):

    """
    Function to get acoustic input features, starting with STFT, needs to be extended to include MFCCs, will ask how many coefficients to use.
    """





    # stft = librosa.core.stft(audio, n_fft = config.nfft, hop_length = config.hopsize, window = config.window).T

    stft = utils.stft(audio, window = config.window, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs)

    assert abs(stft).max() <= 1.0

    # voc_stft_mag = 2 * abs(voc_stft)/np.sum(config.window)

    # voc_stft_phase = np.angle(voc_stft)

    # cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin, bins_per_octave = config.bins_per_octave).T

    # hcqt = get_hcqt(audio)

    # hcqt = np.swapaxes(hcqt, 0,1)

    return stft