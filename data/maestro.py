import torch
from pathlib import Path
import pandas as pd
import random
import time
import librosa
import torchaudio
import pretty_midi
import numpy as np
import re
import soundfile

from data.io import read_single_track_midi, notes_to_rolls_and_events, pedals_to_rolls_and_events, events_to_notes, notes_to_midi, fix_length
from data.tokenizers import Tokenizer


class Maestro:
    def __init__(
        self, 
        root: str = None, 
        split: str = "train",
        segment_seconds: float = 10.,
        tokenizer=None,
        max_token_len=1024,
    ):
    
        self.root = root
        self.split = split
        self.segment_seconds = segment_seconds
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

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
        # t1 = time.time()
        audio_path = Path(self.root, self.audio_filenames[index])
        midi_path = Path(self.root, self.midi_filenames[index]) 
        duration = self.durations[index]

        segment_start_time = random.uniform(0, duration - self.segment_seconds)

        if False:
            index = 0
            index = 27
            audio_path = Path(self.root, self.audio_filenames[0])
            midi_path = Path(self.root, self.midi_filenames[0]) 
            segment_start_time = 12.780630179249533

        # Load audio.
        audio = self.load_audio(audio_path, segment_start_time)
        # shape: (audio_samples)

        # print("a1", time.time() - t1)
        # t1 = time.time()

        # Load tokens.
        string_processor = MaestroStringProcessor(
            label=False,
            onset=True,
            offset=True,
            sustain=False,
            velocity=True,
            pedal_onset=False,
            pedal_offset=False,
            pedal_sustain=False,
        )

        targets_dict = self.load_targets(
            midi_path=midi_path, 
            segment_start_time=segment_start_time, 
            string_processor=string_processor
        )
        # shape: (tokens_num,)

        # librosa.get_samplerate(audio_path)

        data = {
            "audio_path": audio_path,
            "segment_start_time": segment_start_time,
            "audio": audio,
            "frame_roll": targets_dict["frame_roll"],
            "onset_roll": targets_dict["onset_roll"],
            "offset_roll": targets_dict["offset_roll"],
            "velocity_roll": targets_dict["velocity_roll"],
            "event": targets_dict["event"],
            "string": targets_dict["string"],
            "token": targets_dict["token"],
            "tokens_num": targets_dict["tokens_num"],
            "string_processor": targets_dict["string_processor"]
        }


        debug = False
        if debug:
            strings = self.tokenizer.tokens_to_strings(targets_dict["token"])
            events = string_processor.strings_to_events(strings)
            notes = events_to_notes(events)
            notes_to_midi(notes, "_zz.mid")
            soundfile.write(file="_zz.wav", data=audio, samplerate=self.sample_rate)
            from IPython import embed; embed(using=False); os._exit(0)

        # print("a2", time.time() - t1)
        # t1 = time.time()

        return data


    def __len__(self):

        return self.audios_num

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

        audio = np.array(torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=orig_sr, 
            new_freq=self.sample_rate
        ))
        # shape: (audio_samples,)

        return audio

    def load_targets(self, midi_path, segment_start_time, string_processor):

        # Read notes and extend notes by pedal information.
        notes, pedals = read_single_track_midi(midi_path=midi_path, extend_pedal=True)

        seg_start = segment_start_time
        seg_end = seg_start + self.segment_seconds

        label = "maestro-Piano"

        note_data = notes_to_rolls_and_events(
            notes=notes,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps,
            label=label
        )

        pedal_data = pedals_to_rolls_and_events(
            pedals=pedals,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps,
            label=label,
        )

        events = note_data["events"] + pedal_data["events"]
        events.sort(key=lambda event: (event["time"], event["name"]))

        strings = string_processor.events_to_strings(events)

        tokens = self.tokenizer.strings_to_tokens(strings)
        tokens_num = len(tokens)

        tokens = np.array(fix_length(
            x=tokens, 
            max_len=self.max_token_len, 
            constant_value=self.tokenizer.stoi("<pad>")
        ))

        targets_dict = {
            "frame_roll": note_data["frame_roll"],
            "onset_roll": note_data["onset_roll"],
            "offset_roll": note_data["offset_roll"],
            "velocity_roll": note_data["velocity_roll"],
            "ped_frame_roll": pedal_data["frame_roll"],
            "ped_onset_roll": pedal_data["onset_roll"],
            "ped_offset_roll": pedal_data["offset_roll"],
            "event": events,
            "string": strings,
            "token": tokens,
            "tokens_num": tokens_num,
            "string_processor": string_processor
        }

        return targets_dict


class MaestroStringProcessor:
    def __init__(self, 
        label: bool,
        onset: bool, 
        offset: bool, 
        sustain: bool, 
        velocity: bool, 
        pedal_onset: bool, 
        pedal_offset: bool, 
        pedal_sustain: bool, 
    ):
        self.label = label
        self.onset = onset
        self.offset = offset
        self.sustain = sustain
        self.velocity = velocity
        self.pedal_onset = pedal_onset
        self.pedal_offset = pedal_offset
        self.pedal_sustain = pedal_sustain

    def events_to_strings(self, events):

        strings = ["<sos>"]

        for e in events:

            if e["name"] == "note_on":
                if self.onset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])
                    strings = self.append_velocity(strings, e["velocity"])
                
            elif e["name"] == "note_off":
                if self.offset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])

            elif e["name"] == "note_sustain":
                if self.sustain:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])

            elif e["name"] == "pedal_on":
                if self.pedal_onset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])

            elif e["name"] == "pedal_off":
                if self.pedal_offset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])

            elif e["name"] == "pedal_sustain":
                if self.pedal_sustain:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])

            else:
                raise NotImplementedError

        strings.append("<eos>")
        
        return strings

    def strings_to_events(self, strings):

        event = None
        events = []

        for w in strings:

            if "=" in w:
                key = re.search('(.*)=', w).group(1)
                value = re.search('{}=(.*)'.format(key), w).group(1)
                value = self.format_value(key, value)

                if key == "name":
                    if event is not None:
                        events.append(event)
                    event = {}

                event[key] = value

            if w == "<eos>":
                events.append(event)
                break
                

        new_events = []

        for e in events:
            
            if e["name"] == "note_on":

                if not self.velocity:
                    e["velocity"] = 100

                new_events.append(e)

                if not self.offset:
                    event = {
                        "name": "note_off",
                        "time": float(e["time"]) + 0.1,
                        "pitch": e["pitch"]
                    }
                    new_events.append(event)

            elif e["name"] == "note_off":
                new_events.append(e)

        new_events.sort(key=lambda e: (e["time"], e["name"], e["pitch"]))
        
        return new_events


    def append_name(self, strings, name):
        
        strings.append("name={}".format(name))

        return strings

    def append_time(self, strings, time):
        
        strings.append("time={}".format(time))

        return strings

    def append_pitch(self, strings, pitch):
        
        strings.append("pitch={}".format(pitch))

        return strings

    def append_label(self, strings, lab):
        
        if self.label is not None:
            strings.append("label={}".format(lab))

        return strings

    def append_velocity(self, strings, vel):
        
        if self.velocity is not None:
            strings.append("velocity={}".format(vel))

        return strings

    def format_value(self, key, value):
        if key in ["time"]:
            return float(value)

        elif key in ["pitch", "velocity"]:
            return int(value)

        else:
            return value




def test():

    root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"

    
    tokenizer = Tokenizer()

    # Dataset
    dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=10.,
        tokenizer=tokenizer,
        max_token_len=1024,
    )

    data = dataset[0]
    
    audio = data["audio"]
    frame_roll = data["frame_roll"]
    onset_roll = data["onset_roll"]
    offset_roll = data["offset_roll"]
    velocity_roll = data["velocity_roll"]
    tokens = data["token"]
    tokens_num = data["tokens_num"]
    string_processor = data["string_processor"]

    # Convert tokens to strings
    strings = tokenizer.tokens_to_strings(tokens)
    events = string_processor.strings_to_events(strings)
    notes = events_to_notes(events)

    notes_to_midi(notes, "_zz.mid")
    soundfile.write(file="_zz.wav", data=audio, samplerate=16000)

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    test()