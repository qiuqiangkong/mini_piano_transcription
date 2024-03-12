import os
import torch
from pathlib import Path
import pandas as pd
import random
import soundfile
import re
import math
import time
import librosa
import torchaudio
import pretty_midi
import numpy as np
import yaml
import matplotlib.pyplot as plt

from data.tokenizers import Tokenizer
from data.audio_io import AudioIO
from data.io import beats_to_rolls_and_events
from data.io import fix_length, Beat, events_to_beats, add_beats_to_audio
from data.beat_io import BeatStringProcessor


class Ballroom(AudioIO):
    def __init__(
        self, 
        root: str = None, 
        segment_seconds: float = 10.,
        tokenizer=None,
        max_token_len=1024,
    ):
        super(Ballroom, self).__init__()

        self.root = root
        self._segment_seconds = segment_seconds
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        self.sample_rate = 16000
        self.fps = 100

        self.audios_dir = Path(root, "BallroomData")
        self.beats_annot_dir = Path(root, "BallroomAnnotationsBeats")

        meta_csv = Path(self.audios_dir, "allBallroomFiles")

        df = pd.read_csv(meta_csv, header=None)
        df = pd.DataFrame(df)
        self.audio_names = df[0].values

        self.audios_num = len(self.audio_names)

    def __getitem__(self, index):

        audio_path = Path(self.audios_dir, self.audio_names[index])
        beat_annot_path = Path(self.beats_annot_dir, "{}.beats".format(audio_path.stem))
        
        duration = librosa.get_duration(path=audio_path)

        seg_start_time, seg_seconds = self.random_start_time(duration)

        # Load audio.
        audio = self.load_audio(audio_path, seg_start_time, seg_seconds)
        # shape: (audio_samples)

        # Load tokens.
        string_processor = BeatStringProcessor(
            beat=True,
            downbeat=True,
        )

        targets_dict = self.load_beat_targets(
            beat_annot_path=beat_annot_path, 
            segment_start_time=seg_start_time, 
            segment_seconds=seg_seconds,
            string_processor=string_processor
        )
        # shape: (tokens_num,)

        data = {
            "audio_path": audio_path,
            "segment_start_time": seg_start_time,
            "audio": audio,
            "beat_roll": targets_dict["beat_roll"],
            "downbeat_roll": targets_dict["downbeat_roll"],
            "event": targets_dict["event"],
            "string": targets_dict["string"],
            "token": targets_dict["token"],
            "tokens_num": targets_dict["tokens_num"],
            "string_processor": targets_dict["string_processor"]
        }

        return data

    def load_beat_targets(self, beat_annot_path, segment_start_time, segment_seconds, string_processor):

        beats = self.read_beats(beat_annot_path)

        seg_start = segment_start_time
        seg_end = seg_start + segment_seconds
        # seg_frames = round((seg_end - seg_start) * self.fps) + 1

        beat_data = beats_to_rolls_and_events(
            beats=beats,
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps,
        )

        events = beat_data["events"]

        strings = string_processor.events_to_strings(events)

        tokens = self.tokenizer.strings_to_tokens(strings)
        tokens_num = len(tokens)

        tokens = np.array(fix_length(
            x=tokens, 
            max_len=self.max_token_len, 
            constant_value=self.tokenizer.stoi("<pad>")
        ))

        targets_dict = {
            "beat_roll": beat_data["beat_roll"],
            "downbeat_roll": beat_data["downbeat_roll"],
            "event": events,
            "string": strings,
            "token": tokens,
            "tokens_num": tokens_num,
            "string_processor": string_processor
        }

        return targets_dict

    def read_beats(self, beat_annot_path):

        df = pd.read_csv(beat_annot_path, sep=' ', header=None)
        df = pd.DataFrame(df)

        beat_times = df[0].values
        beat_idxes = df[1].values - 1  # Move downbeat to 0

        beats = []

        for beat_time, beat_idx in zip(beat_times, beat_idxes):

            beat = Beat(
                start=beat_time,
                index=beat_idx,
            )
            beats.append(beat)

        return beats


def test():

    root = "/datasets/ballroom"

    tokenizer = Tokenizer()

    # # Dataset
    dataset = Ballroom(
        root=root,
        segment_seconds=10.,
        tokenizer=tokenizer,
    )

    data = dataset[600]

    audio = data["audio"]
    beat_roll = data["beat_roll"]
    downbeat_roll = data["downbeat_roll"]
    tokens = data["token"]
    tokens_num = data["tokens_num"]
    string_processor = data["string_processor"]

    # Convert tokens to strings
    strings = tokenizer.tokens_to_strings(tokens)
    events = string_processor.strings_to_events(strings)
    beats = events_to_beats(events)

    new_audio = add_beats_to_audio(audio, beats, dataset.sample_rate)
    soundfile.write(file="_zz.wav", data=new_audio, samplerate=dataset.sample_rate)

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    test()