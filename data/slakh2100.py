import os
import torch
from pathlib import Path
import pandas as pd
import random
import soundfile
import re
import time
import librosa
import torchaudio
import pretty_midi
import numpy as np
import yaml
import matplotlib.pyplot as plt

from data.tokenizers import Tokenizer
from data.io import read_single_track_midi, notes_to_rolls_and_events, events_to_notes, fix_length, fix_length, mt_notes_to_midi, read_beats, Beat, beats_to_rolls_and_events, events_to_beats, add_beats_to_audio
from data.ballroom import BeatStringProcessor


SLAKH2100_LABELS = [
    "Bass",
    "Brass",
    "Chromatic Percussion",
    "Drums",
    "Ethnic",
    "Guitar",
    "Organ",
    "Percussive",
    "Piano",
    "Pipe",
    "Reed",
    "Sound Effects",
    "Strings",
    "Strings (continued)",
    "Synth Lead",
    "Synth Pad"
]


SLAKH2100_LABELS_TO_MIDI_PROGRAM = {
    "slakh2100-Bass": "Acoustic Bass",
    "slakh2100-Brass": "Trumpet",
    "slakh2100-Chromatic Percussion": "Celesta",
    "slakh2100-Drums": "Drums",
    "slakh2100-Ethnic": "Sitar",
    "slakh2100-Guitar": "Acoustic Guitar (nylon)",
    "slakh2100-Organ": "Drawbar Organ",
    "slakh2100-Percussive": "Tinkle Bell",
    "slakh2100-Piano": "Acoustic Grand Piano",
    "slakh2100-Pipe": "Piccolo",
    "slakh2100-Reed": "Soprano Sax",
    "slakh2100-Sound Effects": "Reverse Cymbal",
    "slakh2100-Strings": "Violin",
    "slakh2100-Strings (continued)": "String Ensemble 1",
    "slakh2100-Synth Lead": "Lead 1 (square)",
    "slakh2100-Synth Pad": "Pad 1 (new age)"
}


class Slakh2100:
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

        self.audios_dir = Path(root, split)
        self.audio_names = sorted(os.listdir(self.audios_dir))

        self.audios_num = len(self.audio_names)


    def __getitem__(self, index):

        target_type = "note"

        song_dir = Path(self.audios_dir, self.audio_names[index])
        audio_path = Path(song_dir, "mix.flac")
        meta_path = Path(song_dir, "metadata.yaml")

        duration = librosa.get_duration(path=audio_path)

        if target_type == "note":

            segment_start_time = random.uniform(0, duration - self.segment_seconds)
            # segment_start_time = 268.5333496526133

            # Load audio.
            audio = self.load_audio(audio_path, segment_start_time)
            # shape: (audio_samples)

            insts_to_load = {label: True for label in SLAKH2100_LABELS}

            # Load tokens.
            string_processor = Slakh2100StringProcessor(
                notes=True,
                drums=True,
                onset=True,
                offset=True,
                sustain=False,
            )

            targets_dict = self.load_targets(
                song_dir=song_dir,
                meta_path=meta_path,
                insts_to_load=insts_to_load,
                segment_start_time=segment_start_time, 
                string_processor=string_processor
            )
            # shape: (tokens_num,)
            
            data = {
                "audio_path": audio_path,
                "segment_start_time": segment_start_time,
                "audio": audio,
                "frame_roll": targets_dict["frame_roll"],
                "onset_roll": targets_dict["onset_roll"],
                "offset_roll": targets_dict["offset_roll"],
                "velocity_roll": targets_dict["velocity_roll"],
                "mt_event": targets_dict["mt_event"],
                "string": targets_dict["string"],
                "token": targets_dict["token"],
                "tokens_num": targets_dict["tokens_num"],
                "string_processor": targets_dict["string_processor"]
            }

        elif target_type == "beat":

            protect_time = 8.
            segment_start_time = random.uniform(protect_time, duration - self.segment_seconds - protect_time)
            # segment_start_time = 268.5333496526133

            # Load audio.
            audio = self.load_audio(audio_path, segment_start_time)
            # shape: (audio_samples)

            # Load tokens.
            string_processor = BeatStringProcessor(
                beat=True,
                downbeat=True,
            )

            midi_path = Path(song_dir, "all_src.mid")

            targets_dict = self.load_beat_targets(
                midi_path=midi_path,
                segment_start_time=segment_start_time, 
                string_processor=string_processor
            )
            # shape: (tokens_num,)

            data = {
                "audio_path": audio_path,
                "segment_start_time": segment_start_time,
                "audio": audio,
                "beat_roll": targets_dict["beat_roll"],
                "downbeat_roll": targets_dict["downbeat_roll"],
                "event": targets_dict["event"],
                "string": targets_dict["string"],
                "token": targets_dict["token"],
                "tokens_num": targets_dict["tokens_num"],
                "string_processor": targets_dict["string_processor"]
            }

        else:
            raise NotImplementedError

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

    def load_targets(self, song_dir, meta_path, insts_to_load, segment_start_time, string_processor):

        stems = self.load_meta(meta_path)

        mt_note_data = {}

        for name, stem in stems.items():

            if stem["audio_rendered"] is True:

                inst_class = stem["inst_class"]
                if inst_class in insts_to_load.keys() and insts_to_load[inst_class] is True:

                    midi_path = Path(song_dir, "MIDI", "{}.mid".format(name)) 
                    
                    notes, _ = read_single_track_midi(midi_path, extend_pedal=True)
                    
                    seg_start = segment_start_time
                    seg_end = seg_start + self.segment_seconds

                    label = "slakh2100-{}".format(stem["inst_class"])
                    
                    note_data = notes_to_rolls_and_events(
                        notes=notes,
                        segment_frames=self.segment_frames, 
                        segment_start=seg_start,
                        segment_end=seg_end,
                        fps=self.fps,
                        label=label,
                    )
                        
                    note_data["is_drum"] = stem["is_drum"]
                    
                    if len(note_data["events"]) > 0:
                        if inst_class not in mt_note_data.keys():
                            mt_note_data[inst_class] = [note_data]
                        else:
                            mt_note_data[inst_class].append(note_data)

        mt_note_data = self.combine_same_insts(mt_note_data)

        # Reduction roll       
        red_frame_roll, red_onset_roll, red_offset_roll = self.multi_tracks_data_to_reduction(mt_note_data
        )

        mt_events = {}

        for key in mt_note_data.keys():
            mt_events[key] = mt_note_data[key]["events"]

        #
        strings = string_processor.mt_note_data_to_strings(mt_note_data)

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
            "mt_event": mt_events,
            "string": strings,
            "token": tokens,
            "tokens_num": tokens_num,
            "string_processor": string_processor
        }

        return targets_dict

    def load_meta(self, meta_path):

        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        stems = meta["stems"]
        return stems

    def combine_same_insts(self, mt_note_data):

        new_mt_note_data = {}

        for inst_class, note_data_list in mt_note_data.items():

            new_mt_note_data[inst_class] = {}

            for key in note_data_list[0].keys():

                if "roll" in key:
                    new_mt_note_data[inst_class][key] = np.clip(np.sum([note_data[key] for note_data in note_data_list], axis=0), a_min=0., a_max=1.)
                    
                elif key == "events":
                    new_mt_note_data[inst_class][key] = []
                    for note_data in note_data_list:
                        new_mt_note_data[inst_class][key].extend(note_data[key])

                    new_mt_note_data[inst_class][key].sort(key=lambda e: (e["time"], e["name"], e["pitch"]))

                elif key == "is_drum":
                    new_mt_note_data[inst_class][key] = note_data_list[0]["is_drum"]

                else:
                    raise NotImplementedError

        return new_mt_note_data

    def multi_tracks_data_to_reduction(self, mt_note_data):

        frame_roll = np.zeros((self.segment_frames, self.pitches_num))
        onset_roll = np.zeros((self.segment_frames, self.pitches_num))
        offset_roll = np.zeros((self.segment_frames, self.pitches_num))

        for note_data in mt_note_data.values():
            if note_data["is_drum"] is False:
                frame_roll += note_data["frame_roll"]
                onset_roll += note_data["onset_roll"]
                offset_roll += note_data["offset_roll"]

        frame_roll = np.clip(frame_roll, a_min=0., a_max=1.)
        onset_roll = np.clip(onset_roll, a_min=0., a_max=1.)
        offset_roll = np.clip(offset_roll, a_min=0., a_max=1.)

        return frame_roll, onset_roll, offset_roll

    def load_beat_targets(self, midi_path, segment_start_time, string_processor):

        beat_times, downbeat_times = read_beats(str(midi_path))
        downbeat_times = list(downbeat_times)
        beats = []

        for beat_time in beat_times:
            if len(downbeat_times) > 0 and beat_time == downbeat_times[0]:
                beat_index = 0
                downbeat_times.pop(0)
            else:
                beat_index += 1

            beat = Beat(start=beat_time, index=beat_index)
            beats.append(beat)

        seg_start = segment_start_time
        seg_end = seg_start + self.segment_seconds

        beat_data = beats_to_rolls_and_events(
            beats=beats,
            segment_frames=self.segment_frames, 
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

    def __len__(self):

        return self.audios_num


class Slakh2100StringProcessor:
    def __init__(self,
        notes: bool,
        drums: bool,
        onset: bool,
        offset: bool,
        sustain: bool,
        # label="maestro-piano"
    ):
        self.notes = notes
        self.drums = drums
        self.onset = onset
        self.offset = offset
        self.sustain = sustain

    def mt_note_data_to_strings(self, mt_note_data):

        all_events = []
        for key, note_data in mt_note_data.items():

            if note_data["is_drum"] is True and self.drums is True:
                all_events.extend(note_data["events"])

            if note_data["is_drum"] is False and self.notes is True:
                all_events.extend(note_data["events"])

        all_events.sort(key=lambda e: (e["time"], e["name"], e["label"], e["pitch"]))
        
        strings = ["<sos>"]

        for e in all_events:

            if e["name"] == "note_on":
                if self.onset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])
                    # strings = self.append_velocity(strings, e["velocity"])
                
            elif e["name"] == "note_off":
                if self.drums and self.offset:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])

            elif e["name"] == "note_sustain":
                if self.drums and self.sustain:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_label(strings, e["label"])
                    strings = self.append_pitch(strings, e["pitch"])

            else:
                raise NotImplementedError

        strings.append("<eos>")

        return strings

    def append_name(self, strings, name):
        
        strings.append("name={}".format(name))

        return strings

    def append_time(self, strings, time):
        
        strings.append("time={}".format(time))

        return strings

    def append_label(self, strings, label):
        
        if label is not None:
            strings.append("label={}".format(label))

        return strings

    def append_pitch(self, strings, pitch):
        
        if pitch is not None:
            strings.append("pitch={}".format(pitch))

        return strings

    def format_value(self, key, value):
        if key in ["time"]:
            return float(value)

        elif key in ["pitch", "velocity"]:
            return int(value)

        else:
            return value

    def strings_to_mt_note_data(self, strings):

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

        # 
        mt_events = {}
        for e in events:
            label = e["label"]
            if label not in mt_events.keys():
                mt_events[label] = [e]
            else:
                mt_events[label].append(e)

        mt_events_full = {key: [] for key in mt_events.keys()}

        for key in mt_events.keys():

            for e in mt_events[key]:
                
                if e["name"] == "note_on":

                    e["velocity"] = 100
                    mt_events_full[key].append(e)

                    if not self.offset:
                        event = {
                            "name": "note_off",
                            "time": float(e["time"]) + 0.01,
                            "pitch": e["pitch"]
                        }
                        mt_events_full[key].append(event)

                elif e["name"] == "note_off":
                    mt_events_full[key].append(e)

            mt_events_full[key].sort(key=lambda e: (e["time"], e["name"], e["pitch"]))
        
        return mt_events_full
        

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

    target_type = "note"
        
    data = dataset[500]

    if target_type == "note":
        
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
        mt_events = string_processor.strings_to_mt_note_data(strings)

        mt_notes = {}
        for key, event in mt_events.items():
            mt_notes[key] = events_to_notes(event)

        mt_notes_to_midi(mt_notes, SLAKH2100_LABELS_TO_MIDI_PROGRAM, "_zz.mid")
        soundfile.write(file="_zz.wav", data=audio, samplerate=16000)

    elif target_type == "beat":

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

    else:
        raise NotImplementedError

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    test()