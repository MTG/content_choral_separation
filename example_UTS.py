# Tests for mutations pertaining to properties.
from synth.models import SIN

model = SIN.SIN()

model.test_file_wav_f0('./Unison_analysis_examples/uni_alto_loc.wav', './Unison_analysis_examples_CREPE/uni_alto_loc_CREPE.f0')

