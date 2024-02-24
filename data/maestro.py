import torch
from pathlib import Path
import pandas as pd
import random
import time
import librosa
import torchaudio
import pretty_midi
import numpy as np

from data.tokenizers import Tokenizer3
from data.midi import read_single_track_midi, notes_to_targets, pedals_to_targets


class Maestro:
    # reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py  
    def __init__(
        self, 
        root: str = None, 
        split: str = "train",
        segment_seconds: float = 10.,
    ):
    
        self.root = root
        self.split = split
        self.segment_seconds = segment_seconds

        self.tokenizer = Tokenizer3()

        self.sample_rate = 16000
        self.fps = 100
        self.pitches_num = 128
        self.segment_frames = int(self.segment_seconds * self.fps) + 1

        self.meta_csv = Path(self.root, "maestro-v2.0.0.csv")

        self.load_meta()
                
    def load_meta(self):

        df = pd.read_csv(self.meta_csv, sep=',')

        indexes = df["split"].values == self.split

        self.midi_filenames = df["midi_filename"].values[indexes]
        self.audio_filenames = df["audio_filename"].values[indexes]
        self.durations = df["duration"].values[indexes]
        self.audios_num = len(self.midi_filenames)

    def __getitem__(self, index):


        # audio_index = random.randint(0, self.audios_num - 1)
        # index = 27

        audio_path = Path(self.root, self.audio_filenames[index])
        midi_path = Path(self.root, self.midi_filenames[index]) 
        duration = self.durations[index]

        segment_start_time = random.uniform(0, duration - self.segment_seconds)

        if False:
            audio_path = Path(self.root, self.audio_filenames[0])
            midi_path = Path(self.root, self.midi_filenames[0]) 
            segment_start_time = 100

        # Load audio.
        audio = self.load_audio(audio_path, segment_start_time)
        # shape: (audio_samples)

        # Load tokens.
        targets_dict = self.load_targets(midi_path, segment_start_time)
        # shape: (tokens_num,)

        # from IPython import embed; embed(using=False); os._exit(0)
        librosa.get_samplerate(audio_path)

        data = {
            "audio_path": audio_path,
            "segment_start_time": segment_start_time,
            "audio": audio,
            # "tokens": targets_dict["tokens"],
            "frame_roll": targets_dict["frame_roll"],
            "onset_roll": targets_dict["onset_roll"],
            "frame_roll": targets_dict["onset_roll"],
            "velocity_roll": targets_dict["velocity_roll"],
        }

        return data


    def __len__(self):

        return self.audios_num

    def load_audio(self, audio_path, segment_start_time):

        maestro_sr = librosa.get_samplerate(audio_path)

        segment_start_sample = int(segment_start_time * maestro_sr)
        segment_samples = int(self.segment_seconds * maestro_sr)

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
            orig_freq=maestro_sr, 
            new_freq=self.sample_rate
        )
        # shape: (audio_samples,)

        return audio

    def load_targets(self, midi_path, segment_start_time):

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

        pedal_data = pedals_to_targets(
            pedals=pedals,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps
        )

        frame_roll = note_data["frame_roll"]
        onset_roll = note_data["onset_roll"]
        offset_roll = note_data["offset_roll"]
        velocity_roll = note_data["velocity_roll"]

        ped_frame_roll = pedal_data["frame_roll"]
        ped_onset_roll = pedal_data["onset_roll"]
        ped_offset_roll = pedal_data["offset_roll"]

        total_events = note_data["events"] + pedal_data["events"]
        total_events.sort(key=lambda event: event["time"])

        # words.append("<eos>")


        # Convert words to tokens.
        # tokens = []
        # for word in words:
        #     token = self.tokenizer.stoi(word)
        #     tokens.append(token)

        targets_dict = {
            # "tokens": tokens,
            "frame_roll": frame_roll,
            "onset_roll": onset_roll,
            "offset_roll": onset_roll,
            "velocity_roll": onset_roll,
        }

        return targets_dict

    '''
    def load_targets(self, midi_path, segment_start_time):

        notes, pedals = read_single_track_midi(midi_path)

        # Load active notes inside the segment.
        active_notes = []
        segment_end_time = segment_start_time + self.segment_seconds
        
        for i in range(len(notes)):
            if segment_start_time <= notes[i].start < segment_end_time:
                active_notes.append(notes[i])

        # Covert notes information to words.
        frames_roll = np.zeros((self.segment_frames, self.pitches_num))
        onsets_roll = np.zeros((self.segment_frames, self.pitches_num))

        words = ["<sos>"]


        for note in active_notes:

            onset_time = note.start - segment_start_time
            offset_time = note.end - segment_start_time
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


        # Convert words to tokens.
        tokens = []
        for word in words:
            token = self.tokenizer.stoi(word)
            tokens.append(token)

        targets_dict = {
            "tokens": tokens,
            "frames_roll": frames_roll,
            "onsets_roll": onsets_roll,
        }

        return targets_dict
    '''

