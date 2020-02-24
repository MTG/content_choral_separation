# Processing the audio for creating the dataset, cuts into segments and adds features like vocoder features, STFT and Tony notes. 
import os

from synth.utils import sig_process, segment, vamp_notes, utils
from synth.config import config

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path, sr=config.fs):
    """
    Placeholder function to load audio from the input file_path. 
    Returns:
        audio: The normalized mono audio wave. 
        fs: The sampling rate.
    """
    audio, fs = librosa.core.load(file_path, sr=config.fs)
    audio = audio/abs(audio).max()
    return audio, fs

def segmenter(audio):
    """
    Segments in in put audio, assumed to be vocals, to phrases. 
    Returns only phrases and not the silences.
    """

    segmenter = segment.VoiceActivityDetection(config.fs, config.silence_ms, 1)

    voc_segments, timestamps = segmenter.process(audio)

    voc_segs = [x for x in voc_segments if x.max()>0.1 and len(x)/config.fs>config.min_phr_len]

    time_out = [y for x,y in zip(voc_segments, timestamps) if x.max()>0.1 and len(x)/config.fs>config.min_phr_len]
    return voc_segs, time_out


def process_seg(audio):
    """
    Process a segment of the audio.
    Returns the world features, TONY annotated notes and the STFT.
    """
    out_feats = sig_process.get_world_feats(audio)
    #Test if the reverse works.
    # audio_out = sig_process.feats_to_audio(out_feats)

    traj = vamp_notes.extract_notes_pYIN_vamp(audio)

    # timestamps = np.arange(0, float(traj[-1][1]), config.hoptime)
    timestamps = np.arange(0, len(out_feats)*config.hoptime, config.hoptime)

    out_notes = vamp_notes.note2traj(traj, timestamps)

    out_notes = sig_process.f0_to_hertz(out_notes)

    out_notes[out_notes== -np.inf] = 0

    out_stft = abs(np.array(utils.stft(audio, hopsize=config.hopsize, nfft=config.framesize, fs=config.fs)))

    out_feats, out_notes, out_stft = utils.match_time([out_feats, out_notes, out_stft])

    assert all(out_feats[:,-2]>0)

    assert len(out_feats) == len(out_notes)

    return out_feats, out_notes, out_stft

def process_audio(audio):
    """
    Process an audio input. cuts into segments and returns the required features. 
    """

    segments, timestamps = segmenter(audio)
    time_outs = [(x[0], x[-1]) for x in timestamps]
    out_features = []
    out_notes = []
    out_stfts = []
    for segment in segments:
        segment_features, segment_notes, segment_stft = process_seg(segment)
        out_features.append(segment_features)
        out_notes.append(segment_notes)
        out_stfts.append(segment_stft)
    return segments, time_outs, np.array(out_features), np.array(out_notes), np.array(out_stfts)


