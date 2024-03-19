import numpy as np
import re
import math

from data.io import time_to_grid


class Beat:
    def __init__(self, start, index):
        self.start = start
        self.index = index

    def __repr__(self):
        return "Beat(start={}, index={})".format(self.start, self.index)


class BeatStringProcessor:
    def __init__(self, 
        beat: bool,
        downbeat: bool,
        beat_index: bool
    ):
        self.beat = beat
        self.beat_index = beat_index
        self.downbeat = downbeat
        
    def events_to_strings(self, events):

        strings = ["<sos>"]

        for e in events:

            if e["name"] == "beat":
                if self.beat:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    if self.beat_index:
                        strings = self.append_beat_index(strings, e["index"])

            elif e["name"] == "downbeat":
                if self.downbeat:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])

            else:
                raise NotImplementedError

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

                if key == "name":
                    event = {key: value}
                    event, shift = self.look_forward(strings, i, event)
                    events.append(event)
                    i += shift
                    continue

            i += 1

        return events

    def look_forward(self, strings, i, event):

        for j in range(1, 10):
            if strings[i + j] == "<eos>":
                return event, j

            next_key = re.search('(.*)=', strings[i + j]).group(1)
            next_value = re.search('=(.*)', strings[i + j]).group(1)
            next_value = self.format_value(next_key, next_value)

            if next_key == "name":
                return event, j
            else:
                event[next_key] = next_value


    def append_name(self, strings, name):
        
        strings.append("name={}".format(name))

        return strings

    def append_time(self, strings, time):
        
        strings.append("time={}".format(time))

        return strings

    def append_beat_index(self, strings, beat_index):
        
        strings.append("beat_index={}".format(beat_index))

        return strings

    def format_value(self, key, value):
        if key in ["time"]:
            return float(value)

        elif key in ["beat_index"]:
            return int(value)

        else:
            return value


def beats_to_rolls_and_events(
    beats,
    segment_start, 
    segment_end, 
    fps
):
    seg_start = segment_start
    seg_end = segment_end
    seg_frames = round((seg_end - seg_start) * fps) + 1
    
    beat_roll = np.zeros(seg_frames)
    downbeat_roll = np.zeros(seg_frames)
    events = []

    for beat in beats:

        beat_time = beat.start - seg_start
        beat_time = time_to_grid(beat_time, fps)

        if seg_start <= beat.start <= seg_end:

            frame_idx = round(beat_time * fps)
            beat_roll[frame_idx] += 1

            events.append({
                "name": "beat",
                "time": beat_time,
                "index": beat.index
            })

            if beat.index == 0:
                downbeat_roll[frame_idx] += 1

                events.append({
                    "name": "downbeat",
                    "time": beat_time,
                })

    # No need to sort events because they are already sorted.

    data = {
        "beat_roll": beat_roll,
        "downbeat_roll": downbeat_roll,
        "events": events,
    }

    return data


def events_to_beats(events):

    beats = []

    for e in events:
        
        if e["name"] == "beat":
            if "beat_index" in e.keys():
                beat = Beat(start=e["time"], index=e["beat_index"])
            else:
                beat = Beat(start=e["time"], index=None)
            beats.append(beat)
    
    return beats


def add_beats_to_audio(audio, beats, sample_rate):

    audio_samples = audio.shape[-1]
    new_audio = np.copy(audio)

    for beat in beats:

        n = np.arange(sample_rate / 10)
        r = (2 ** (1. / 12))
        if beat.index is None:
            beat_index = 0
        else:
            beat_index = beat.index
        freq = 880 * (r ** (- beat_index))
        beat_seg = np.cos(2 * math.pi * freq / sample_rate * n)
        bgn = int(beat.start * sample_rate)
        end = bgn + len(n)
        
        if end > audio_samples:
            end = audio_samples
            beat_seg = beat_seg[0 : end - bgn]

        new_audio[bgn : end] += beat_seg

    return new_audio