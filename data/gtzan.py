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


LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", 
    "pop", "reggae", "rock"]
LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}
CLASSES_NUM = len(LABELS)


class Gtzan(AudioIO):
    def __init__(
        self, 
        root: str = None, 
        split: str = "train",
        fold: int = 0,
        segment_seconds: float = 10.,
        tokenizer=None,
        max_token_len=1024,
    ):
        super(Gtzan, self).__init__()

        self.root = root
        self.split = split
        self.fold = fold
        self._segment_seconds = segment_seconds
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        self.sample_rate = 16000
        self.fps = 100
        self.classes_num = CLASSES_NUM

        self.meta_dict = self.load_meta()
        self.audios_num = len(self.meta_dict["audio_name"])

    def load_meta(self):

        audios_dir = Path(self.root, "genres")

        genres = sorted(os.listdir(audios_dir))

        meta_dict = {
            "label": [],
            "audio_name": [],
            "audio_path": []
        }

        for genre in genres:

            audio_names = sorted(os.listdir(Path(audios_dir, genre)))

            train_audio_names, test_audio_names = self.split_train_test(audio_names)

            if self.split == "train":
                filtered_audio_names = train_audio_names
            elif self.split == "test":
                filtered_audio_names = test_audio_names
            elif self.split == "full":
                filtered_audio_names = train_audio_names + test_audio_names
            else:
                raise NotImplementedError

            for audio_name in filtered_audio_names:

                audio_path = Path(audios_dir, genre, audio_name)

                meta_dict["label"].append(genre)
                meta_dict["audio_name"].append(audio_name)
                meta_dict["audio_path"].append(audio_path)

        return meta_dict

    def split_train_test(self, audio_names):

        train_audio_names = []
        test_audio_names = []

        test_ids = range(self.fold * 10, (self.fold + 1) * 10)

        for audio_name in audio_names:

            audio_id = int(re.search(r'\d+', audio_name).group())

            if audio_id in test_ids:
                test_audio_names.append(audio_name)
            else:
                train_audio_names.append(audio_name)

        return train_audio_names, test_audio_names

    def __getitem__(self, index):

        audio_path = self.meta_dict["audio_path"][index]
        
        duration = librosa.get_duration(path=audio_path)

        seg_start_time, seg_seconds = self.random_start_time(duration)

        # Load audio.
        audio = self.load_audio(audio_path, seg_start_time, seg_seconds)
        # shape: (audio_samples)

        target_type = "tag"

        if target_type == "tag":

            label = self.meta_dict["label"][index]

            # Load tokens.
            string_processor = LabelStringProcessor()

            targets_dict = self.load_tag_targets(
                label=label,
                string_processor=string_processor
            )
            # shape: (tokens_num,)

            data = {
                "audio_path": audio_path,
                "segment_start_time": seg_start_time,
                "audio": audio,
                "event": targets_dict["event"],
                "string": targets_dict["string"],
                "token": targets_dict["token"],
                "tokens_num": targets_dict["tokens_num"],
                "string_processor": targets_dict["string_processor"]
            }

        elif target_type == "beat":

            beat_name = "gtzan_" + audio_path.stem.replace(".", "_") + ".beats"
            beat_annot_path = Path(self.root, "gtzan_tempo_beat/beats", beat_name)

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

        else:
            raise NotImplementedError

        return data

    def load_beat_targets(self, 
        beat_annot_path, 
        segment_start_time, 
        segment_seconds, 
        string_processor
    ):

        beats = self.load_beats(beat_annot_path)

        seg_start = segment_start_time
        seg_end = seg_start + segment_seconds

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

    def load_beats(self, beat_annot_path):
        

        df = pd.read_csv(beat_annot_path, sep='\t', header=None)
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

    def load_tag_targets(self, label, string_processor):

        target = np.zeros(self.classes_num)
        class_index = LB_TO_IX[label]
        target[class_index] = 1

        # events = beat_data["events"]
        events = [{"label": "gtzan-{}".format(label)}]

        strings = string_processor.events_to_strings(events)

        tokens = self.tokenizer.strings_to_tokens(strings)
        tokens_num = len(tokens)

        tokens = np.array(fix_length(
            x=tokens, 
            max_len=self.max_token_len, 
            constant_value=self.tokenizer.stoi("<pad>")
        ))

        targets_dict = {
            "event": events,
            "string": strings,
            "token": tokens,
            "tokens_num": tokens_num,
            "string_processor": string_processor
        }

        return targets_dict


class LabelStringProcessor:
    def __init__(self):
        pass

    def events_to_strings(self, events):

        strings = ["<sos>"]
        for e in events:
            strings.append("label={}".format(e["label"]))
        strings.append("<eos>")
        
        return strings

    def strings_to_events(self, strings):

        events = []

        i = 0

        while i < len(strings):

            if "=" in strings[i]:
                key = re.search('(.*)=', strings[i]).group(1)
                value = re.search('=(.*)', strings[i]).group(1)
                value = self.format_value(key, value)

                event = {key: value}
                events.append(event)

            i += 1

        return events

    def format_value(self, key, value):
        
        return value


def test():

    root = "/datasets/gtzan"

    tokenizer = Tokenizer()

    # # Dataset
    dataset = Gtzan(
        root=root,
        # segment_seconds=10.,
        segment_seconds=None,
        tokenizer=tokenizer,
    )

    data = dataset[101]

    target_type = "tag"

    if target_type == "tag":

        audio = data["audio"]
        tokens = data["token"]
        tokens_num = data["tokens_num"]
        string_processor = data["string_processor"]

        # Convert tokens to strings
        strings = tokenizer.tokens_to_strings(tokens)
        events = string_processor.strings_to_events(strings)

        soundfile.write(file="_zz.wav", data=audio, samplerate=dataset.sample_rate)

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