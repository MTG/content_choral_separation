from synth.config import config
import numpy as np
import librosa
import pyworld as pw
from synth.utils.reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp, mgc_to_mfsc, mfsc_to_mgc
import sys
import os,re
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import synth.utils.segment
import scipy
from tqdm import tqdm
from resemblyzer import preprocess_wav, VoiceEncoder
import librosa


def get_embedding_GE2E(filename):

    wav, _ = librosa.load(str(filename), sr=22050)
    encoder = VoiceEncoder(device="cpu")
    emb = encoder.embed_utterance(wav)
    return emb

def griffinlim(spectrogram, n_iter = 50, window = 'hann', n_fft = 1024, hop_length = -1, verbose = False):

    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        # full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = istft(spectrogram,angles, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)
        rebuilt = stft(inverse, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)[:spectrogram.shape[0],:]
        angles = np.exp(1j * np.angle(rebuilt))
        progress(i,n_iter)
        # import pdb;pdb.set_trace()

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    # full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = istft(spectrogram, angles, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

    return inverse


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def stft(data, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow

    # import pdb;pdb.set_trace()
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data

    # import pdb;pdb.set_trace()
    if len(data.shape)>1:
        data = np.mean(data, axis = -1)
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    
    # zero-padding data such that it holds an exact number of frames

    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = n*hopsize
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[int(beginFrame):int(endFrame)]
        STFT[int(n),:] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")
        
    # frequency and time stamps:
    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT
def istft(mag, phase, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0,
          analysisWindow=None):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    """
    X = mag * np.exp(1j*phase)
    X = X.T
    if analysisWindow is None:
        analysisWindow = window

    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)

    normalisationSeq = np.zeros(lengthData)

    data = np.zeros(lengthData)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft), norm = 'ortho')
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)

    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.

    data = data / normalisationSeq

    return data
def melspectrogram(y):
    S = _amp_to_db(_linear_to_mel(np.abs(y))) - config.ref_level_db
    # if not config.allow_clipping_in_normalization:
    # assert S.max() <= 0 and S.min() - config.min_level_db >= 0
    return _normalize(S)

def _linear_to_mel(spectrogram):
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    assert config.f_max <= config.fs // 2
    return librosa.filters.mel(config.fs, config.nfft,
                               fmin=config.f_min, fmax=config.f_max,
                               n_mels=config.num_mels)


def _amp_to_db(x):
    min_level = np.exp(config.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - config.min_level_db) / -config.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -config.min_level_db) + config.min_level_db


def stft_to_feats(vocals, fs = config.fs):
    if len(vocals.shape)>1:
        vocals = vocals[:,0]
        vocals = np.ascontiguousarray(vocals)

    feats=pw.wav2world(vocals,fs = config.fs,frame_period=config.hoptime*1000)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10 * np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    harm = harm - 20
    f0 = feats[0]

    is_voiced = f0 > 0.0
    if not np.any(is_voiced):
        pass  # all unvoiced, do nothing
    else:
        for k in range(ap.shape[1]):
            ap[~is_voiced, k] = np.interp(np.where(~is_voiced)[0], np.where(is_voiced)[0], ap[is_voiced, k])
    # f0_1 = pitch.extract_f0_sac(vocals, fs, 0.00580498866)

    

    y=69+12*np.log2(f0/440)
    # y = hertz_to_new_base(f0)
    nans, x= nan_helper(y)
    naners=np.isinf(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)

    harm = np.nan_to_num(harm)

    ap = np.nan_to_num(ap)

    harmy= sp_to_mfsc(harm+1e-12,60,0.45)
    apy= ap_to_wbap(ap+1e-12,4,config.fs)

    


    out_feats=np.concatenate((harmy,apy,y.reshape((-1,2))),axis=1) 

    # import pdb;pdb.set_trace()

    # audio_out = feats_to_audio(out_feats)

    # sf.write('./test_mfsc.wav', audio_out, config.fs)



    # import pdb;pdb.set_trace()

    # harm_in=mgc_to_sp(harmy, 1025, 0.45)
    # ap_in= wbap_to_ap(apy, 1025, config.fs)

    # harm_in = 10**((harm_in + 20)/10)
    # ap_in = np.clip(10**(ap_in/20), 0.0, 1.0)

    # audio_out = pw.synthesize(f0 , np.ascontiguousarray(harm_in).astype('double') , np.ascontiguousarray(ap).astype('double'),config.fs,config.hoptime*1000)

    # sf.write('./test.wav', audio_out, config.fs)

    return out_feats, f0



def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def generate_overlapadd(allmix,time_context=config.max_phr_len, overlap=config.max_phr_len/2,batch_size=config.batch_size):
    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    input_size = allmix.shape[-1]

    i=0
    start=0  
    while (start + time_context) < allmix.shape[0]:
        i = i + 1
        start = start - overlap + time_context 
    fbatch = np.zeros([int(np.ceil(float(i)/batch_size)),batch_size,time_context,input_size])+1e-10
    
    
    i=0
    start=0  

    while (start + time_context) < allmix.shape[0]:
        fbatch[int(i/batch_size),int(i%batch_size),:,:]=allmix[int(start):int(start+time_context),:]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    
    return fbatch,i

def overlapadd(fbatch,nchunks,overlap=int(config.max_phr_len/2)):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[1]


    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    

    sep = np.zeros((int(nchunks*(time_context-overlap)+time_context),input_size))

    
    i=0
    start=0 
    while i < nchunks:
        #import pdb;pdb.set_trace()
        s = fbatch[int(i/batch_size),int(i%batch_size),:,:]

        #print s1.shape
        if start==0:
            sep[0:time_context] = s

        else:
            #print start+overlap
            #print start+time_context
            sep[int(start+overlap):int(start+time_context)] = s[overlap:time_context]
            sep[start:int(start+overlap)] = window[overlap:]*sep[start:int(start+overlap)] + window[:overlap]*s[:overlap]
        i = i + 1 #index for each block
        start = int(start - overlap + time_context) #starting point for each block
    return sep  

def match_time(feat_list):
    """ 
    Matches the shape across the time dimension of a list of arrays.
    Assumes that the first dimension is in time, preserves the other dimensions
    """
    shapes = [f.shape for f in feat_list]
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[0] for s in shapes])
        new_list = []
        for i in range(len(feat_list)):
            new_list.append(feat_list[i][:min_time])
        feat_list = new_list
    return feat_list

def query_yes_no(question, default="yes"):
    """
    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
def monolize(audio):
    if len(audio.shape) == 2:
      vocals = np.array((audio[:,1]+audio[:,0])/2)
    else: 
      vocals = np.array(audio)
    return vocals

def process_f0(f0, f_bins, n_freqs):
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))

    haha = np.digitize(f0, f_bins) - 1

    idx2 = haha < n_freqs

    haha = haha[idx2]

    freqz[range(len(haha)), haha] = 1

    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T

    min_target = np.min(atb[range(len(haha)), haha])

    atb = atb / min_target

    atb[atb > 1] = 1

    return atb

def slice_data(song_dir, voc_wav_file, back_wav_file):
    voc_audio,fs = librosa.load(os.path.join(song_dir, voc_wav_file), sr = config.fs)
    voc_audio = monolize(voc_audio)

    back_audio, fs = librosa.load(os.path.join(song_dir, back_wav_file), sr = config.fs)
    back_audio = monolize(back_audio)

    voc_stft = stft(voc_audio, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window )

    energy = librosa.feature.rmse(S=voc_stft.T)

    # import pdb;pdb.set_trace()

    first_idx = np.argmax(energy.T>config.energy_threshold) - 2

    last_idx = len(energy.T) - np.argmax(energy.T[::-1]>config.energy_threshold) + 2

    back_audio = back_audio[first_idx*config.hopsize: last_idx*config.hopsize]

    voc_audio = voc_audio[first_idx*config.hopsize: last_idx*config.hopsize]

    voc_audio, back_audio = match_time([voc_audio, back_audio])

    segmenter = segment.VoiceActivityDetection(fs, config.ms_sil, 1)    

    voc_segments, back_segments = segmenter.process(voc_audio, back_audio)

    voc_segments = [x for x in voc_segments if len(x)>config.hopsize*config.max_phr_len]

    back_segments = [x for x in back_segments if len(x)>config.hopsize*config.max_phr_len]

    return voc_segments, back_segments


def list_to_file(in_list,filename):
    filer=open(filename,'w')
    for jj in in_list:
        filer.write(str(jj)+'\n')
    filer.close()

def process_data(voc_audio, back_audio, f_bins, n_freqs):

    voc_stft = stft(voc_audio, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

    # mix_audio = (voc_audio + back_audio)/2

    # mix_stft = utils.stft(mix_audio, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

    back_stft = stft(back_audio, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

    voc_mel = melspectrogram(voc_stft.T).T

    # mix_mel = utils.melspectrogram(mix_stft.T).T

    world_feats, f0 = stft_to_feats(voc_audio.astype('double'))

    atb = process_f0(f0, f_bins, n_freqs)


    voc_stft, voc_mel, world_feats, back_stft, atb = match_time([voc_stft, voc_mel, world_feats, back_stft, atb])

    return voc_stft, voc_mel, world_feats, back_stft, atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins
def one_hotize(inp, max_index):


    output = np.eye(max_index)[inp.astype(int)]

    return output
def get_multif0(pitch_activation_mat, freq_grid, time_grid, thresh=0.3):
    """Compute multif0 output containing all peaks in the output that
       fall above thresh
    Parameters
    ----------
    pitch_activation_mat : np.ndarray
        Deep salience prediction
    freq_grid : np.ndarray
        Frequency values
    time_grid : np.ndarray
        Time values
    thresh : float, default=0.3
        Likelihood threshold
    Returns
    -------
    times : np.ndarray
        Time values
    freqs : list
        List of lists of frequency values
    """
    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(time_grid))]
    # import pdb;pdb.set_trace()
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freq_grid[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return time_grid, est_freqs

def process_output(atb):
    freq_grid = librosa.cqt_frequencies(config.cqt_bins, config.fmin, config.bins_per_octave)
    time_grid = np.linspace(0, config.hoptime * atb.shape[0], atb.shape[0])
    time_grid, est_freqs = get_multif0(atb.T, freq_grid, time_grid)
    return time_grid, est_freqs

def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])
