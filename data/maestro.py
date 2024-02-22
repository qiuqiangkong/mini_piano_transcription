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


        audio_index = random.randint(0, self.audios_num - 1)
        audio_index = 27

        audio_path = Path(self.root, self.audio_filenames[audio_index])
        midi_path = Path(self.root, self.midi_filenames[audio_index]) 
        duration = self.durations[audio_index]

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
            "audio": audio,
            "tokens": targets_dict["tokens"],
            "frames_roll": targets_dict["frames_roll"],
            "onsets_roll": targets_dict["onsets_roll"],
            "audio_path": audio_path,
            "segment_start_time": segment_start_time,
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

        midi_data = pretty_midi.PrettyMIDI(str(midi_path))

        assert len(midi_data.instruments) == 1

        notes = midi_data.instruments[0].notes
        control_changes = midi_data.instruments[0].control_changes
        
        # Get pedals.
        pedals = get_pedals(control_changes)

        # Extend note offsets by pedal information.
        notes = extend_offset_by_pedal(notes, pedals)

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


def get_pedals(control_changes):

    onset = None
    offset = None
    pairs = []

    control_changes.sort(key=lambda cc: cc.time)

    for cc in control_changes:

        if cc.number == 64:

            if cc.value >= 64 and onset is None:
                onset = cc.time

            elif cc.value < 64 and onset is not None:
                offset = cc.time
                pairs.append((onset, offset))
                onset = None
                offset = None

    if onset is not None and offset is None:
        offset = control_changes[-1].time
        pairs.append((onset, offset))

    return pairs


def extend_offset_by_pedal(notes, pedals):

    notes.sort(key=lambda note: note.end)

    notes_dict = {pitch: [] for pitch in range(128)}

    while len(pedals) > 0 and len(notes) > 0:

        ped_on, ped_off = pedals[0]

        while notes:

            note = notes[0]
            note_on = note.start
            note_off = note.end
            pitch = note.pitch
            velocity = note.velocity

            if note_off < ped_on:
                notes_dict[pitch].append(note)
                notes.pop(0)

            elif ped_on <= note_off < ped_off:

                new_note = pretty_midi.Note(
                    pitch=pitch, 
                    start=note_on, 
                    end=ped_off, 
                    velocity=velocity
                )

                if len(notes_dict[pitch]) > 0:
                    if notes_dict[pitch][-1].end > new_note.start:
                        notes_dict[pitch][-1].end = new_note.start
                notes_dict[pitch].append(new_note)
                notes.pop(0)

            elif ped_off <= note_off:
                pedals.pop(0)
                break

            else:
                raise NotImplementedError 

    for note in notes:
        notes_dict[note.pitch].append(note)

    new_notes = []
    for pitch in notes_dict.keys():
        new_notes.extend(notes_dict[pitch])
    
    new_notes.sort(key=lambda note: note.start)

    return new_notes