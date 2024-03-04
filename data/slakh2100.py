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
from data.io import read_single_track_midi, notes_to_rolls_and_events#, drums_to_rolls_and_events


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
        self.max_token_len = max_token_len

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
            segment_start_time=segment_start_time, 
            string_processor=string_processor
        )
        # shape: (tokens_num,)
        # shape: (tokens_num,)

        stems = self.load_meta(meta_path)

        # mt_note_data = {}

        # for name, stem in stems.items():

        #     if stem["audio_rendered"] is True:

        #         midi_path = Path(song_dir, "MIDI", "{}.mid".format(name)) 
        #         # note_data = self.add(midi_path, segment_start_time)

        #         notes, pedals = read_single_track_midi(midi_path)
                
        #         note_data = notes_to_targets(
        #             notes=notes,
        #             segment_frames=self.segment_frames, 
        #             segment_start=segment_start_time,
        #             segment_end=segment_start_time + self.segment_seconds,
        #             fps=self.fps
        #             # label=
        #         )

        #         from IPython import embed; embed(using=False); os._exit(0)

        #         note_data["slakh2100_inst_class"] = stem["inst_class"]
        #         note_data["slakh2100_is_drum"] = stem["is_drum"]

        #         mt_note_data[name] = note_data

        # mt_events = self.multi_tracks_data_to_events(mt_note_data)

        # mt_events.sort(key=lambda event: (event["time"], event["name"]))

        # red_frame_roll, red_onset_roll, red_offset_roll = self.multi_tracks_data_to_reduction(mt_note_data)

        # # fig, axs = plt.subplots(2,1, sharex=True)
        # # axs[0].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet')
        # # axs[1].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet')
        # # plt.savefig("_zz.pdf")

        # # soundfile.write(file="_zz.wav", data=audio, samplerate=self.sample_rate)

        # # Load tokens.
        # # targets_dict = self.load_targets(midi_path, segment_start_time)
        # # shape: (tokens_num,)

        # from IPython import embed; embed(using=False); os._exit(0)

        # data = {
        #     "audio": audio,
        #     "frame_roll": red_frame_roll,
        #     "onset_roll": red_onset_roll,
        #     "offset_roll": red_offset_roll,
        # }

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

    def load_targets(self, song_dir, meta_path, segment_start_time, string_processor):

        stems = self.load_meta(meta_path)

        mt_note_data = {}

        for name, stem in stems.items():

            if stem["audio_rendered"] is True:

                midi_path = Path(song_dir, "MIDI", "{}.mid".format(name)) 
                
                notes, _ = read_single_track_midi(midi_path, extend_pedal=True)
                
                seg_start = segment_start_time
                seg_end = seg_start + self.segment_seconds

                label = "slakh2100-{}".format(stem["midi_program_name"])
                is_drum = stem["is_drum"]

                # if is_drum:
                #     note_data = drums_to_rolls_and_events(
                #         notes=notes,
                #         segment_frames=self.segment_frames, 
                #         segment_start=seg_start,
                #         segment_end=seg_end,
                #         fps=self.fps,
                #         label=label,
                #     )
                # else:
                note_data = notes_to_rolls_and_events(
                    notes=notes,
                    segment_frames=self.segment_frames, 
                    segment_start=seg_start,
                    segment_end=seg_end,
                    fps=self.fps,
                    label=label,
                )
                    
                note_data["is_drum"] = is_drum
                
                mt_note_data[name] = note_data
         
        # Reduction roll       
        red_frame_roll, red_onset_roll, red_offset_roll = self.multi_tracks_data_to_reduction(mt_note_data
        )

        #
        strings = string_processor.mt_note_data_to_strings(mt_note_data)

        from IPython import embed; embed(using=False); os._exit(0)

        mt_events = self.multi_tracks_data_to_events(mt_note_data)


        # Read notes and extend notes by pedal information.
        notes, pedals = read_single_track_midi(midi_path=midi_path, extend_pedal=True)

        seg_start = segment_start_time
        seg_end = seg_start + self.segment_seconds

        note_data = notes_to_rolls_and_events(
            notes=notes,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps
        )

        pedal_data = pedals_to_rolls_and_events(
            pedals=pedals,
            segment_frames=self.segment_frames, 
            segment_start=seg_start,
            segment_end=seg_end,
            fps=self.fps
        )

        events = note_data["events"] + pedal_data["events"]
        events.sort(key=lambda event: (event["time"], event["name"]))

        strings = string_processor.events_to_strings(events)

        tokens = self.tokenizer.strings_to_tokens(strings)

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
            "string_processor": string_processor
        }

        return targets_dict

    def load_meta(self, meta_path):

        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        stems = meta["stems"]
        return stems

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


    # def multi_tracks_data_to_events(self, mt_note_data):

    #     mt_events = []

    #     for note_data in mt_note_data.values():
    #         mt_events.extend(note_data["events"])

    #     return mt_events

    


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

        strings = ["<sos>"]

        for key, note_data in mt_note_data.items():

            if note_data["is_drum"] is True:
                if self.drums is False:
                    continue
            else:
                if self.notes is False:
                    continue

            for e in note_data["events"]:

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

    def append_label(self, strings, label):
        
        if label is not None:
            strings.append("label={}".format(label))

        return strings

    def append_pitch(self, strings, pitch):
        
        if pitch is not None:
            strings.append("pitch={}".format(pitch))

        return strings

    # def append_velocity(self, strings, vel):
        
    #     if self.velocity is not None:
    #         strings.append("velocity={}".format(vel))

    #     return strings

    def format_value(self, key, value):
        if key in ["time"]:
            return float(value)

        elif key in ["pitch", "velocity"]:
            return int(value)

        else:
            return value



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