import os
import torch
import time
import pretty_midi
import copy
import yaml
from pathlib import Path

from data.maestro import Maestro
from data.slakh2100 import Slakh2100
from data.collate import collate_fn
from train import Sampler
# from data.midi import read_beats, beats_to_targets


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

    len(set(inst_names))
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


if __name__ == '__main__':
    add3()