import os
import torch
import time
import pretty_midi
import copy
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import math
import soundfile
from scipy import signal
import matplotlib.pyplot as plt

from data.maestro import Maestro
# from data.slakh2100 import Slakh2100
from data.collate import collate_fn
from train import Sampler 
# from data.beat_io import read_beats, beats_to_targets


def add():

    t1 = time.time()
    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes
    print(time.time() - t1)


def add2():

    root = "/datasets/slakh2100_flac"

    dataset = Slakh2100(root=root)

    # Sampler
    sampler = Sampler(dataset_size=len(dataset))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    for data in dataloader:
        pass


def load_meta(meta_path):

    with open(meta_path, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    stems = meta["stems"]
    return stems


def add3():

    audios_dir = "/datasets/slakh2100_flac/train"

    audio_names = sorted(os.listdir(audios_dir))

    inst_names = []

    for audio_name in audio_names:
        meta_path = Path(audios_dir, audio_name, "metadata.yaml")

        load_meta(meta_path)
        
        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        for stem_name in meta["stems"].keys():
            inst_name = meta["stems"][stem_name]["inst_class"]
            inst_names.append(inst_name)

    unique_names = set(inst_names)
    for name in unique_names:
        print(name, np.sum(np.array(inst_names) == name))
    
    from IPython import embed; embed(using=False); os._exit(0)


def _sub(midi_path):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    # assert len(midi_data.instruments) == 1

    # notes = midi_data.instruments[0].notes
    # control_changes = midi_data.instruments[0].control_changes
    
    print(midi_data.get_tempo_changes())
    beats = midi_data.get_beats()
    downbeats = midi_data.get_downbeats()

    from IPython import embed; embed(using=False); os._exit(0) 

    # return notes, pedals


def add4():

    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    # _sub(midi_path)

    # root = "/datasets/slakh2100_flac/train"
    # audio_names = sorted(os.listdir(root))

    # for audio_name in audio_names:
    #     print(audio_name)
    #     song_dir = Path(root, audio_name)
    #     audio_path = Path(song_dir, "mix.flac")
    #     meta_path = Path(song_dir, "metadata.yaml")
    #     midi_path = Path(song_dir, "all_src.mid")

    #     _sub(midi_path)





def add5():

    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    t1 = time.time()
    beats, downbeats = read_beats(midi_path)

    segment_frames = 1001
    seg_start = 25.3
    seg_end = 35.
    fps = 100

    data = beats_to_targets(beats, downbeats, segment_frames, seg_start, seg_end, fps)
    print(time.time() - t1) 

    from IPython import embed; embed(using=False); os._exit(0)


def add6():

    sample_rate = 16000

    dataset_root = "/datasets/ballroom"
    audios_dir = Path(dataset_root, "BallroomData")
    beats_annot_dir = Path(dataset_root, "BallroomAnnotationsBeats")

    meta_csv = Path(dataset_root, "BallroomData/allBallroomFiles")


    df = pd.read_csv(meta_csv, header=None)
    df = pd.DataFrame(df)
    audio_names = df[0].values

    for audio_name in audio_names:
        audio_path = Path(audios_dir, audio_name)
        audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)

        beat_annot_path = Path(beats_annot_dir, "{}.beats".format(Path(audio_name).stem))

        df = pd.read_csv(beat_annot_path, sep=' ', header=None)
        df = pd.DataFrame(df)

        beat_times = df[0].values
        beat_ids = df[1].values

        tmp = np.array(audio)
        # tmp = np.zeros_like(audio)
        # for i in range(len(beat_times)):
        for beat_time, beat_id in zip(beat_times, beat_ids):
            n = np.arange(1600)
            a1 = np.cos(2 * math.pi * 440 / sample_rate * n)
            bgn = int(beat_time * sample_rate)
            end = bgn + len(n)
            tmp[bgn : end] += a1

        soundfile.write(file="_zz.wav", data=tmp, samplerate=sample_rate)

        from IPython import embed; embed(using=False); os._exit(0)

        
def add7():

    for i in range(100):
        print(i)
        i += 10


def add8():

    a1 = np.load("/datasets/harmonix/melspecs/0001_12step-mel.npy")

    import matplotlib.pyplot as plt
    plt.matshow(np.log(a1), origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add9():
    from datasets import load_dataset

    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")

    print(gigaspeech)


def add10():

    sos = signal.iirfilter(
        N=17, 
        # Wn=[0.3, 0.5], 
        Wn=[0.5],
        rs=60, 
        # btype='band',
        btype='lowpass',
        analog=False, 
        ftype='cheby2', 
        fs=2000,
        output='sos'
    )
    w, h = signal.sosfreqz(sos, 2000, fs=2000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.abs(h))
    # ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    ax.set_title('Chebyshev Type II bandpass frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    # ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add11():

    b, a = signal.iirfilter(
        N=17, 
        Wn=[2*np.pi*50, 2*np.pi*500], 
        rs=60,
        btype='band', 
        analog=True, 
        ftype='cheby2'
    )
    w, h = signal.freqs(b, a, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    # ax.plot(np.abs(h))
    ax.set_title('Chebyshev Type II bandpass frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    # ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    plt.show()
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add12():

    b, a = signal.iirfilter(
        N=20, 
        Wn=[500],
        # Wn=[2*np.pi*1334.24],
        rs=60,
        btype='lowpass', 
        analog=False, 
        ftype='cheby2',
        fs=2000
    )
    w, h = signal.freqs(b, a, 2000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    # ax.plot(np.abs(h))
    # ax.set_title('Chebyshev Type II bandpass frequency response')
    # ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel('Amplitude [dB]')
    # ax.axis((10, 1000, -100, 10))
    # ax.grid(which='both', axis='both')
    plt.show()
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add13():

    import numpy as np
    import scipy.signal

    np.random.seed(42)  # for reproducibility
    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds
    ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr

    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, yraw)

    from matplotlib import pyplot as plt

    plt.figure(figsize=[6.4, 2.4])

    plt.plot(ts, yraw, label="Raw signal")
    plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")

    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
               ncol=2, fontsize="smaller")

    plt.savefig("_zz.pdf")


def add14():

    import numpy as np
    import scipy.signal

    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds
    ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr

    b, a = scipy.signal.iirfilter(N=4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    print(b, a)
    y_lfilter = scipy.signal.lfilter(b, a, yraw)

    from matplotlib import pyplot as plt
    plt.plot(ts, yraw, label="Raw signal")
    plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")

    plt.savefig("_zz.pdf")

    w, h = signal.freqs(b, a, fs)

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
    add14()