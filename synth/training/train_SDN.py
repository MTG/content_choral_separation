# Tests for mutations pertaining to properties.
import os

from synth.models import SDN
from synth.config import config

# waves, feats, notes, phons, songs = prepare_nus.walk_directory("ADIZ")
def train():
    model = SDN.SDN()
    model.train()
