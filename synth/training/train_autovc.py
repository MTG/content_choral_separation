# Tests for mutations pertaining to properties.
import os

from synth.models import autovc

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def train():
    model = autovc.AutoVC()
    model.train()
