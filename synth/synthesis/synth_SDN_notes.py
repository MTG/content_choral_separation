# Tests for mutations pertaining to properties.
import os

from synth.models import SDN_notes

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def synthesize_hdf5(filename, singer_index):
    model = SDN_notes.SDN()
    model.test_file_hdf5(filename, singer_index)

def synthesize_wav(filename, singer_index):
    model = SDN_notes.SDN()
    model.test_file_wav(filename, singer_index)
