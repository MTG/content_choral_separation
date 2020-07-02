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


def segmenter_yam(audio, audio_back):
    """
    Segments in in put audio, assumed to be vocals, to phrases. 
    Returns only phrases and not the silences.
    """

    segmenter = segment.VoiceActivityDetectionYAM(config.fs, config.silence_ms, 1)

    voc_segments, back_segments = segmenter.process(audio, audio_back)



    voc_segs = []

    back_segs = []

    # i = 0

    for  x, y in zip(voc_segments, back_segments):
        if abs(x).mean()>0.015 and len(x)/config.fs>config.min_phr_len:
            # sf.write('./tests/test_{}.wav'.format(i), x, config.fs)
            # i+=1
            voc_segs.append(x)
            back_segs.append(y)

    return voc_segs, back_segs

def process_seg(audio):
    """
    Process a segment of the audio.
    Returns the world features, TONY annotated notes and the STFT.
    """
    out_feats = sig_process.get_world_feats(audio)
    #Test if the reverse works.
    # audio_out = sig_process.feats_to_audio(out_feats)

    traj = vamp_notes.extract_notes_pYIN_vamp(audio)

    if traj.shape[0]<1 or len(out_feats)<=config.max_phr_len:
        return None,None,None
    else:

        timestamps = np.arange(0, float(traj[-1][1]), config.hoptime)

        out_notes = vamp_notes.note2traj(traj, timestamps)

        out_notes_1 = sig_process.f0_to_hertz(out_notes[:,0])

        out_notes_1[out_notes_1== -np.inf] = 0

        out_notes[:,0] = out_notes_1

        out_stft = abs(np.array(utils.stft(audio, hopsize=config.hopsize, nfft=config.framesize, fs=config.fs)))

        out_feats, out_notes, out_stft = utils.match_time([out_feats, out_notes, out_stft])

        if len(out_feats)<=config.max_phr_len:
            return None,None,None
        else:

            assert all(out_feats[:,-2]>0)

            assert len(out_feats) == len(out_notes)

            return out_feats, out_notes, out_stft


def process_seg_yam(audio, audio_back):
    """
    Process a segment of the audio.
    Returns the world features, TONY annotated notes and the STFT.
    """
    out_feats = sig_process.get_world_feats(audio)
    #Test if the reverse works.
    # audio_out = sig_process.feats_to_audio(out_feats)

    traj = vamp_notes.extract_notes_pYIN_vamp(audio)

    if traj.shape[0]<1 or len(out_feats)<=config.max_phr_len:
        return None,None,None
    else:

        timestamps = np.arange(0, float(traj[-1][1]), config.hoptime)

        out_notes = vamp_notes.note2traj(traj, timestamps)

        out_notes_1 = sig_process.f0_to_hertz(out_notes[:,0])

        out_notes_1[out_notes_1== -np.inf] = 0

        out_notes[:,0] = out_notes_1

        out_stft = abs(np.array(utils.stft(audio, hopsize=config.hopsize, nfft=config.framesize, fs=config.fs)))
        back_stft = abs(np.array(utils.stft(audio_back, hopsize=config.hopsize, nfft=config.framesize, fs=config.fs)))

        out_feats, out_notes, out_stft, back_stft = utils.match_time([out_feats, out_notes, out_stft, back_stft])

        if len(out_feats)<=config.max_phr_len:
            return None,None,None, None
        else:

            assert all(out_feats[:,-2]>0)

            assert len(out_feats) == len(out_notes)

            return out_feats, out_notes, out_stft, back_stft


def process_audio(audio):
    """
    Process an audio input. cuts into segments and returns the required features. 
    """

    segments, timestamps = segmenter(audio)
    time_outs = [(x[0], x[-1]) for x in timestamps]
    out_features = []
    out_notes = []
    out_stfts = []
    for times, segment in zip(time_outs, segments):
        segment_features, segment_notes, segment_stft = process_seg(segment)
        if segment_features is not None:
            out_features.append(segment_features)
            out_notes.append(segment_notes)
            out_stfts.append(segment_stft)
    return segments, time_outs, np.array(out_features), np.array(out_notes), np.array(out_stfts)

def process_audio_yam(audio, audio_back):
    """
    Process an audio input. cuts into segments and returns the required features. 
    """

    segments, back_segments = segmenter_yam(audio, audio_back)
    out_features = []
    out_notes = []
    out_stfts = []
    back_stfts = []
    for back_segment, segment in zip(back_segments, segments):
        segment_features, segment_notes, segment_stft, back_stft = process_seg_yam(segment, back_segment)
        if segment_features is not None:
            out_features.append(segment_features)
            out_notes.append(segment_notes)
            out_stfts.append(segment_stft)
            back_stfts.append(back_stft)
    return segments, np.array(back_stfts), np.array(out_features), np.array(out_notes), np.array(out_stfts)

