# Tests for mutations pertaining to properties.
import os

from synth.models import autovc_notes

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def train():
    model = autovc_notes.AutoVC()
    model.train()
