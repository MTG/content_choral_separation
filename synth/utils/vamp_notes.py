import soundfile as sf
import librosa
import vamp
import numpy as np
import matplotlib.pyplot as plt
from synth.utils import segment

def extract_notes_pYIN_vamp(x, Fs=22050, H=128, N=1024):
    # pYIN parameters
    param = {'threshdistr': 2, 'outputunvoiced': 2, 'precisetime': 0}
    # Options: smoothedpitchtrack, f0candidates, f0probs, voicedprob, candidatesalience, smoothedpitchtrack, notes
    pYIN_note_output = vamp.collect(x, Fs, "pyin:pyin", output='notes', parameters=param, step_size=H, block_size=N)
    # reformating
    traj = np.empty((0, 3))
    for entry in pYIN_note_output['list']:
        timestamp = float(entry['timestamp'])
        duration = float(entry['duration'])
        note = float(entry['values'][0])
        traj = np.vstack((traj, [timestamp, timestamp+duration, note]))
    return traj


def extract_F0_pYIN_vamp(x, Fs=22050, H=128, N=1024):
    # pYIN parameters
    param = {'threshdistr': 2, 'outputunvoiced': 2, 'precisetime': 0}
    # Options: smoothedpitchtrack, f0candidates, f0probs, voicedprob, candidatesalience, smoothedpitchtrack, notes
    pYIN_f0_output = vamp.collect(x, Fs, "pyin:pyin", output='smoothedpitchtrack', parameters=param, step_size=H, block_size=N)

    return pYIN_f0_output['vector'][1]

    # reformating
    # traj = np.empty((0, 3))
    # import pdb;pdb.set_trace()
    # for entry in pYIN_f0_output['list']:
    #     timestamp = float(entry['timestamp'])
    #     duration = float(entry['duration'])
    #     note = float(entry['values'][0])
    #     traj = np.vstack((traj, [timestamp, timestamp+duration, note]))
    # return traj

def note2traj(note_info, timebase):
    traj = np.hstack((timebase.reshape(-1, 1), np.zeros(len(timebase)).reshape(-1, 1)))
    for i in range(note_info.shape[0]):
        # get indices of trajectory
        t_start_idx = np.argmin(np.abs(timebase - note_info[i, 0]))
        t_end_idx = np.argmin(np.abs(timebase - note_info[i, 1]))
        note_len_idx = t_end_idx - t_start_idx
        traj[t_start_idx:t_end_idx, 1] = note_info[i, 2]
    return traj[:,1]




