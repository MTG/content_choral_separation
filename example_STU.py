from synth.models import autovc_emb

model = autovc_emb.AutoVC()
model.solo_unison_file_wav('./Unison_analysis_examples/sol_soprano1_loc.wav', std=0.5, num_singers=4, timing=5)

