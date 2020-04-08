# Tests for mutations pertaining to properties.
import os

from synth.models import SDN_notes
from synth.config import config

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def train():
    model = SDN_notes.SDN()
    model.train()
