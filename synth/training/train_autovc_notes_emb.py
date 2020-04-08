# Tests for mutations pertaining to properties.
import os

from synth.models import autovc_notes_emb

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def train():
    model = autovc_notes_emb.AutoVC()
    model.train()
