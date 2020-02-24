# Tests for mutations pertaining to properties.
import os

from synth.models import autovc

def synthesize_hdf5(filename, singer_index):
    """
    Function to load the AutoVC model and test on an HDF5 file.
    """
    model = autovc.AutoVC()
    model.test_file_hdf5(filename, singer_index)