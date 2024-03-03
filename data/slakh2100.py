import os
import torch
from pathlib import Path
import pandas as pd
import random
import soundfile
import time
import librosa
import torchaudio
import pretty_midi
import numpy as np
import yaml
import matplotlib.pyplot as plt

from data.tokenizers import Tokenizer
from data.io import read_single_track_midi, notes_to_targets, pedals_to_targets


class Slakh2100:
    # reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py  
    def __init__(
        self, 
        root: str = None, 
        split: str = "train",
        segment_seconds: float = 10.,
        tokenizer=None,
    ):
    
        self.root = root
        self.split = split
        self.segment_seconds = segment_seconds

        self.sample_rate = 16000
        self.fps = 100
        self.pitches_num = 128
        self.segment_frames = int(self.segment_seconds * self.fps) + 1

        self.audios_dir = Path(root, split)
        self.audio_names = sorted(os.listdir(self.audios_dir))

        self.audios_num = len(self.audio_names)


    def __getitem__(self, index):


        # audio_index = random.randint(0, self.audios_num - 1)
        audio_index = 0

        song_dir = Path(self.audios_dir, self.audio_names[index])
        audio_path = Path(song_dir, "mix.flac")
        meta_path = Path(song_dir, "metadata.yaml")

        duration = librosa.get_duration(path=audio_path)

        segment_start_time = random.uniform(0, duration - self.segment_seconds)

        # Load audio.
        audio = self.load_audio(audio_path, segment_start_time)
        # shape: (audio_samples)

        stems = self.load_meta(meta_path)

        mt_note_data = {}

        for name, stem in stems.items():

            if stem["audio_rendered"] is True:

                midi_path = Path(song_dir, "MIDI", "{}.mid".format(name)) 
                # note_data = self.add(midi_path, segment_start_time)

                notes, pedals = read_single_track_midi(midi_path)
                from IPython import embed; embed(using=False); os._exit(0)

                note_data = notes_to_targets(
                    notes=notes,
                    segment_frames=self.segment_frames, 
                    segment_start=segment_start_time,
                    segment_end=segment_start_time + self.segment_seconds,
                    fps=self.fps
                )

                note_data["slakh2100_inst_class"] = stem["inst_class"]
                note_data["slakh2100_is_drum"] = stem["is_drum"]

                mt_note_data[name] = note_data

        mt_events = self.multi_tracks_data_to_events(mt_note_data)

        mt_events.sort(key=lambda event: (event["time"], event["name"]))

        red_frame_roll, red_onset_roll, red_offset_roll = self.multi_tracks_data_to_reduction(mt_note_data)

        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet')
        # axs[1].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        # soundfile.write(file="_zz.wav", data=audio, samplerate=self.sample_rate)

        # Load tokens.
        # targets_dict = self.load_targets(midi_path, segment_start_time)
        # shape: (tokens_num,)

        from IPython import embed; embed(using=False); os._exit(0)

        data = {
            "audio": audio,
            "frame_roll": red_frame_roll,
            "onset_roll": red_onset_roll,
            "offset_roll": red_offset_roll,
        }

        return data

    def load_audio(self, audio_path, segment_start_time):

        orig_sr = librosa.get_samplerate(audio_path)

        segment_start_sample = int(segment_start_time * orig_sr)
        segment_samples = int(self.segment_seconds * orig_sr)

        audio, fs = torchaudio.load(
            audio_path, 
            frame_offset=segment_start_sample, 
            num_frames=segment_samples
        )
        # (channels, audio_samples)

        audio = torch.mean(audio, dim=0)
        # shape: (audio_samples,)

        audio = torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=orig_sr, 
            new_freq=self.sample_rate
        )
        # shape: (audio_samples,)

        return audio

    def load_meta(self, meta_path):

        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        stems = meta["stems"]
        return stems

    def add(self, midi_path, segment_start_time):

        notes, pedals = read_single_track_midi(midi_path)

        seg_start = segment_start_time
        seg_end = seg_start + self.segment_seconds

        note_data = notes_to_targets(
            notes=notes,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps
        )
        from IPython import embed; embed(using=False); os._exit(0)

        return note_data


    def multi_tracks_data_to_events(self, mt_note_data):

        mt_events = []

        for note_data in mt_note_data.values():
            mt_events.extend(note_data["events"])

        return mt_events

    def multi_tracks_data_to_reduction(self, all_note_data):

        pitches_num = 128
        frame_roll = np.zeros((self.segment_frames, pitches_num))
        onset_roll = np.zeros((self.segment_frames, pitches_num))
        offset_roll = np.zeros((self.segment_frames, pitches_num))

        for note_data in all_note_data.values():
            if note_data["slakh2100_is_drum"] is False:
                frame_roll += note_data["frame_roll"]
                onset_roll += note_data["onset_roll"]
                offset_roll += note_data["offset_roll"]

        frame_roll = np.clip(frame_roll, a_min=0., a_max=1.)
        onset_roll = np.clip(onset_roll, a_min=0., a_max=1.)
        offset_roll = np.clip(offset_roll, a_min=0., a_max=1.)

        return frame_roll, onset_roll, offset_roll


    def __len__(self):

        return self.audios_num


def test():

    root = "/datasets/slakh2100_flac"

    tokenizer = Tokenizer()

    # Dataset
    dataset = Slakh2100(
        root=root,
        split="train",
        segment_seconds=10.,
        tokenizer=tokenizer,
    )

    data = dataset[0]

    tokens = data["token"]

    # Convert tokens to words
    words = tokens_to_words(tokens, tokenizer)

    # TODO
    # words_to_events()

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    test()