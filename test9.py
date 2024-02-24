import torch
import time
import pretty_midi
import copy
import numpy as np
import math

from data.maestro import Maestro
from data.slakh2100 import Slakh2100
from data.collate import collate_fn
from data.midi import read_single_track_midi, write_notes_to_midi
from train import Sampler


def add():
    root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"

    dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=10.,
    )

    sampler = Sampler(dataset_size=len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    for _ in range(100):
        # t1 = time.time()
        data = dataset[0]
        # print(time.time() - t1)

    # for data in dataloader:
    #     from IPython import embed; embed(using=False); os._exit(0)

    

###
def add4():

    
    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    notes, pedals = read_single_track_midi(midi_path)

    write_notes_to_midi(notes, "_zz.mid")

    segment_frames = 1001
    seg_start = 25.
    seg_end = 35.
    fps = 100

    note_data = notes_to_targets(
        notes=notes,
        segment_frames=segment_frames, 
        segment_start=seg_start,
        segment_end=seg_end,
        fps=fps
    )

    pedal_data = pedals_to_targets(
        pedals=pedals,
        segment_frames=segment_frames, 
        segment_start=seg_start,
        segment_end=seg_end,
        fps=fps
    )

    from IPython import embed; embed(using=False); os._exit(0)

    words = ["<sos>"]


    for note in active_notes:

        onset_time = note.start - segment_start
        offset_time = note.end - segment_end
        pitch = note.pitch
        velocity = note.velocity

        words.append("<time>={}".format(onset_time))
        words.append("<pitch>={}".format(pitch))
        words.append("<velocity>={}".format(velocity))

        onset_index = int(np.round(onset_time * self.fps))
        offset_index = int(np.round(offset_time * self.fps))
        frames_roll[onset_index : min(self.segment_frames, offset_index + 1), pitch] = 1
        onsets_roll[onset_index, pitch] = 1

    words.append("<eos>")





if __name__ == '__main__':
    add4()