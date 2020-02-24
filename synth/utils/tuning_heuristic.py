import numpy as np

import scipy.signal

from frontend.acoustic.basic import smooth
from frontend.acoustic.pitch import read_tools_f0, write_tools_f0, hz_to_midi_note, midi_note_to_hz
from frontend.control import htk_lab_file, notes_file

import matplotlib.pyplot as plt  # just for plots


def main():
    fn_base = './data/triangler'
    master_tuning = 0.0  # cents
    
    # filenames
    fn_tools_f0 = fn_base + '.tools.f0'
    fn_lab      = fn_base + '.lab'
    fn_notes    = fn_base + '.notes'  # XXX: these should be notes with times modified by note timing model!!! maybe add check that sees if vowel onset matches note onsets
    fn_out_tools_f0 = fn_base + '.tools.f0.tuned'

    # read f0
    f0, hoptime = read_tools_f0(fn_tools_f0)

    f0_orig = np.array(f0, copy=True)  # XXX: just for plotting

    # read lab
    phn_items = htk_lab_file.read(fn_lab)

    # read notes
    notes = notes_file.read_old_fmt(fn_notes)

    # check note onsets match vowel onsets
    # i.e. we should use note times after they were modified by note timing model
    raise NotImplementedError  # todo

    # compute smoothed derivate
    assert np.all(f0 > 0), 'expected continuous input pitch'
    f0 = hz_to_midi_note(f0)

    # 50ms derivative filter length, rounded up to next odd integer
    wl = int(np.round(50.0e-3/hoptime))
    wl = wl + 1 - (wl % 2)

    # derivative by sinusoidal FIR filter
    #deriv_filter = -np.sin(np.linspace(0, 1, wl)*2*np.pi)/13.5  # correlate
    #deriv_filter = np.sin(np.linspace(0, 1, wl)*2*np.pi)/13.5  # convolve
    #assert wl == 11, '13.5 factor is tuned for wl=11 only; use Savitzky-Golay filter for any wl'

    deriv_filter = scipy.signal.savgol_coeffs(wl, polyorder=3, deriv=1, use='conv')

    #df0 = np.correlate(f0, deriv_filter, 'same')
    df0 = np.convolve(f0, deriv_filter, 'same')

    #wf0 = 1 / (1 + np.abs(df0)/50)
    ## XXX: is /50 too much? wf0 (after smoothing is like 0.94 ~ 1.0)
    #sl = int(np.round(150e-3/hoptime))
    #from frontend.acoustic.basic import smooth_moving
    #wf0 = smooth_moving(wf0, sl)
    # XXX: wf0 is not used!


    if 0:
        deriv_filter2 = scipy.signal.savgol_coeffs(wl, polyorder=3, deriv=1, use='conv')
        
        #deriv_filter2 /= np.sqrt(np.mean(deriv_filter2**2))  # XXX: not norm, but RMS normalize

        df0_2 = np.convolve(f0, deriv_filter2, 'same')

        #df0_2 /= np.sqrt(np.mean(deriv_filter2**2))
        df0_2 /= np.linalg.norm(deriv_filter2)/np.sqrt(wl)

        ax = plt.subplot(2, 1, 1)
        ax.plot(deriv_filter)
        ax.plot(deriv_filter2)

        ax = plt.subplot(2, 1, 2)
        ax.plot(df0)
        ax.plot(df0_2)

        plt.show()
        return


    # frame-wise weighting based on phonetics
    # i.e. when computing average f0 along a note, 
    # weight vowels and syllabic consonants more than consonants,
    # silences are not considered alltogether
    L = len(f0)
    w_phn = np.ones(L)
    for b_time, e_time, phn in phn_items:
        b = int(np.round(b_time/hoptime))
        e = int(np.round(e_time/hoptime))
        b = np.clip(b, 0, L-1)
        e = np.clip(e, 0, L)
        if phn in ['sil', 'pau', 'br']:
            w_phn[b:e] = 0.0
        elif phn in ['a', 'e', 'i', 'o', 'u', 'N']:
            w_phn[b:e] = 2.0

    if 0:
        plt.plot(w_phn)
        plt.show()


    # add sub-note segments, besides notes
    # ...
    # XXX: alternatively, do two passes; 1) notes, 2) sub-note segments

    # compute transposition
    def approx_segment_avg_f0(f0, df0, w_phn, f0_tar, b, e):
        # window to reduce influence of edges of note (25% fade in/out)
        n = e - b
        w_e = scipy.signal.tukey(n, 0.5)

        # weighting depending on derivative (stable f0 is weighted higher); range: 1/15 (big derivative) - 1 (zero derivative, flat pitch)
        w_d = 1/np.clip(1 + 27*np.abs(df0[b:e]), None, 15)

        if 0:
            ax = plt.subplot(2, 1, 1)
            ax.plot(13.5*np.abs(df0[b:e]))
            ax = plt.subplot(2, 1, 2)
            ax.plot(w_d)
            plt.show()
            import sys
            sys.exit()

        # weighting depending on phonetic regions (vowels and syllabic consonants are weighted higher than consonants, silences are excluded)
        if np.sum(w_phn[b:e]) > 0:  # avoid zero weighting
            w_p = w_phn[b:e]
        else:
            w_p = 1

        # weighting depending on difference from target (big deviations from target are weighted less); range: ~1/24 (big deviation, e.g. 2 octaves) - 1 (<= +/- 1 semitone deviation)
        w_t = 1/np.clip(np.abs(f0[b:e] - f0_tar), 1, None)

        # weighted average
        w = w_e*w_d*w_p*w_t
        avg_f0_segment = np.sum(f0[b:e]*w)/np.sum(w)

        return avg_f0_segment

    transp = np.zeros(L)

    # 1. over note segments
    for note in notes:
        if note.is_rest:
            continue

        f0_tar = note.note_num

        b = int(np.round(note.b_time/hoptime))
        e = int(np.round(note.e_time/hoptime))
        b = np.clip(b, 0, L-1)
        e = np.clip(e, 0, L)

        # estimate average f0 along note segment
        avg_f0_note = approx_segment_avg_f0(f0, df0, w_phn, f0_tar, b, e)

        # compute difference (with optional master tuning to adjust target)
        transp[b:e] = f0_tar + master_tuning/100 - avg_f0_note

    # 2. additionally, for long notes, over sub-note segments
    for note in notes:
        if note.is_rest:
            continue

        f0_tar = note.note_num

        dur = note.e_time - note.b_time

        if dur >= 1.25:
            len_seg = 0.2  # 20% duration per segment
            hop_seg = 0.1  # 50% overlap
            
            p_segs = np.arange(0.0, 1.0 - len_seg, hop_seg)  # begin points of 9 segments

            b_per_seg = []  # begin (frames)
            e_per_seg = []  # end (frames)
            transp_per_seg = []  # transposition (semitones)

            # segment transpositions
            for p in p_segs:
                b = note.b_time + p*dur
                e = b + len_seg*dur
                b = int(np.round(b/hoptime))
                e = int(np.round(e/hoptime))
                b = np.clip(b, 0, L-1)
                e = np.clip(e, 0, L)

                # estimate average f0 along sub-note segment
                avg_f0_note = approx_segment_avg_f0(f0, df0, w_phn, f0_tar, b, e)

                # compute difference (with optional master tuning to adjust target)
                transp_seg = f0_tar + master_tuning/100 - avg_f0_note
                
                b_per_seg.append(b)
                e_per_seg.append(e)
                transp_per_seg.append(transp_seg)

            # linearly interpolate transposition
            # NOTE: this is a little different from original JB algorithm
            b = int(np.round(note.b_time/hoptime))
            e = int(np.round(note.e_time/hoptime))
            b = np.clip(b, 0, L-1)
            e = np.clip(e, 0, L)

            n_segs = len(transp_per_seg)
            transp_interp = np.empty(n_segs + 2)
            transp_interp[0]    = transp_per_seg[0]  # left edge (flat first segment)
            transp_interp[1:-1] = transp_per_seg[:]
            transp_interp[-1]   = transp_per_seg[-1]  # right edge (flat last segment)

            time_interp = np.empty(n_segs + 2, dtype=np.int)
            time_interp[0]    = b  # left edge
            time_interp[1:-1] = np.round((np.array(b_per_seg) + np.array(e_per_seg)) / 2).astype(np.int)  # segment centers
            time_interp[-1]   = e  # right edge

            transp0 = np.array(transp[b:e], copy=True)  # for plotting only
            transp[b:e] = np.interp(np.arange(b, e), time_interp, transp_interp)

            if 1:
                ax = plt.subplot(1, 1, 1)
                ax.plot(np.arange(b, e), f0[b:e], '-g', alpha=0.4)
                ax.plot(np.arange(b, e), f0_tar + transp0, ':k')
                ax.plot(np.arange(b, e), f0_tar + transp[b:e], '--b')
                ax.plot(time_interp, f0_tar + transp_interp, 'or')
                plt.show()
                #return


    # smooth transposition
    stransp = smooth(transp, hoptime, 150e-3, kernel='gaussian')

    # apply transposition
    tf0 = f0 + stransp
    f0 = midi_note_to_hz(tf0)

    if 1:
        ax = plt.subplot(1, 1, 1)
        ax.plot(f0_orig)
        ax.plot(f0, '--')
        plt.show()


    # save f0 file
    write_tools_f0(fn_out_tools_f0, f0, hoptime)

    print('OK')


if __name__ == '__main__':
    main()