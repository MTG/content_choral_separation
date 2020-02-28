# Code for processing midi anf lab files

import os,re
import numpy as np
import vamp
import re
import matplotlib.pyplot as plt
from scipy.stats import norm

from synth.config import config
from synth.utils import sig_process, segment, vamp_notes

def coarse_code(x, n_states = 3, sigma = 0.4):
    """Coarse-code value to finite number of states, each with a Gaussian response.

    Parameters
    ----------
    x : ndarray
        Vector of normalized values [0.0;1.0], shape (nframes,).
    n_states : int
        Number of states to use for coase coding.
    sigma : float
        Sigma (scale, standard deviation) parameter of normal distribution 
        used internally to perform coarse coding. Default: 0.4

    Returns
    -------
    ndarray
        Matrix of shape (nframes, n_states).

    See also
    --------
    https://en.wikipedia.org/wiki/Neural_coding#Position_coding
    https://plus.google.com/+IlyaEdrenkin/posts/B55jf3wUBvD
    https://github.com/CSTR-Edinburgh/merlin/blob/master/src/frontend/label_normalisation.py
    """
    assert np.all(x >= 0.0) and np.all(x <= 1.0), 'expected input to be normalized in range [0;1]'
    mu = np.linspace(0.0, 1.0, num=n_states, endpoint=True)
    return np.hstack([norm.pdf(x, mu_k, sigma).reshape((-1, 1)) for mu_k in mu]) 


def note_str_to_num(note, base_octave=-1):
    """Convert note pitch as string to MIDI note number."""
    patt = re.match('^([CDEFGABcdefgab])([b#]*)(-?)(\d+)$', note)
    if patt is None:
        raise ValueError('invalid note string "{}"'.format(note))
    base_map = {'C': 0,
                'D': 2,
                'E': 4,
                'F': 5,
                'G': 7,
                'A': 9,
                'B': 11}
    base, modifiers, sign, octave = patt.groups()
    base_num = base_map[base.upper()]
    mod_num = -modifiers.count('b') + modifiers.count('#')
    sign_mul = -1 if sign == '-' else 1
    octave_num = 12*int(octave)*sign_mul - 12*base_octave
    note_num = base_num + mod_num + octave_num
    if note_num < 0 or note_num >= 128:
        raise ValueError('note string "{}" resulted in out-of-bounds note number {:d}'.format(note, note_num))
    return note_num


def note_num_to_str(note, base_octave=-1):
    """Convert MIDI note number to note pitch as string."""
    base = note % 12
    # XXX: base_map should probably depend on key
    base_map = ['C',
                'C#',
                'D',
                'D#',
                'E',
                'F',
                'F#',
                'G',
                'G#',
                'A',
                'A#',
                'B']
    base_note = note%12
    octave = int(np.floor(note/12)) + base_octave
    return '{}{:d}'.format(base_map[base_note], octave)

def pho_to_segment(phos, start_time, end_time):
    """
    Process a sequence of phonemes with start time and endtimes to start and end time of the corresponding segment. 
    """
    return vamp_notes.note2traj(phos, np.arange(start_time, end_time, config.hoptime))

def pho_segment_allign(phos, timestamps):
    out_phos = []
    for int_count, timestamp in enumerate(timestamps):
        out_pho = pho_to_segment(phos, timestamp[0], timestamp[1])
        out_phos.append(out_pho)
    return out_phos


def open_lab_file(filename):
    """
    Returns a numpy array with the start-time, end-time and phonemes from the lab file
    """
    with open(filename, "r") as lab_f:
        phos = lab_f.readlines()
        phonemas = config.phonemas
        phos2 = [x.split() for x in phos]
        phos3 = np.array([[float(x[0]), float(x[1]), phonemas.index(x[2])] for x in phos2])
    return phos3

def open_f0_file(filename):
    """
    Returns a numpy array with the start-time, end-time and notes from the f0 file
    """
    with open(filename, "r") as lab_f:
        phos = lab_f.readlines()
        phos2 = [x.split() for x in phos]
        phos3 = np.array([[float(x[0]), float(x[0]) + 0.005804988, float(x[1])] for x in phos2])
    return phos3