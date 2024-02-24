import time
import copy
import math
import numpy as np

import pretty_midi


class Pedal:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def read_single_track_midi(midi_path):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    assert len(midi_data.instruments) == 1

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes
    
    # Get pedals.
    pedals = get_pedals(control_changes)
    
    # Extend note offsets by pedal information.
    notes = extend_offset_by_pedal(notes, pedals)

    return notes, pedals


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
                pedal = Pedal(start=onset, end=offset)
                pairs.append(pedal)
                onset = None
                offset = None

    if onset is not None and offset is None:
        offset = control_changes[-1].time
        pedal = Pedal(start=onset, end=offset)
        pairs.append(pedal)

    return pairs


def extend_offset_by_pedal(notes, pedals):

    notes = copy.deepcopy(notes)
    pedals = copy.deepcopy(pedals)
    pitches_num = 128

    notes.sort(key=lambda note: note.end)

    notes_dict = {pitch: [] for pitch in range(pitches_num)}

    while len(pedals) > 0 and len(notes) > 0:

        pedal = pedals[0]

        while notes:

            note = notes[0]
            pitch = note.pitch
            velocity = note.velocity

            if 0 <= note.end < pedal.start:
                notes_dict[pitch].append(note)
                notes.pop(0)

            elif pedal.start <= note.end < pedal.end:

                new_note = pretty_midi.Note(
                    pitch=pitch, 
                    start=note.start, 
                    end=pedal.end, 
                    velocity=velocity
                )

                if len(notes_dict[pitch]) > 0:
                    if notes_dict[pitch][-1].end > new_note.start:
                        notes_dict[pitch][-1].end = new_note.start
                notes_dict[pitch].append(new_note)
                notes.pop(0)

            elif pedal.end <= note.end < math.inf:
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


def write_notes_to_midi(notes, midi_path):

    midi_data = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)
    for note in notes:
        track.notes.append(note)
    midi_data.instruments.append(track)
    midi_data.write(midi_path)
    print("Write MIDI to {}".format(midi_path))


def notes_to_targets(notes, segment_frames, segment_start, segment_end, fps):

    #
    seg_start = segment_start
    seg_end = segment_end
    pitches_num = 128

    # Covert notes information to words.
    frame_roll = np.zeros((segment_frames, pitches_num))
    onset_roll = np.zeros((segment_frames, pitches_num))
    offset_roll = np.zeros((segment_frames, pitches_num))
    velocity_roll = np.zeros((segment_frames, pitches_num))

    events = []

    for note in notes:

        onset_time = note.start - seg_start
        offset_time = note.end - seg_start
        pitch = note.pitch
        velocity = note.velocity
        
        if 0 <= note.start < seg_start and seg_start <= note.end < seg_end:

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx, pitch] += 1
            frame_roll[0 : offset_idx + 1, pitch] += 1

            events.append({
                "name": "note_sustain", 
                "time": 0, 
                "pitch": pitch, 
                "velocity": velocity
            })
            events.append({
                "name": "note_off",
                "time": offset_time, 
                "pitch": pitch,
            })

        if 0 <= note.start < seg_start and seg_end <= note.end < math.inf:

            frame_roll[:, pitch] += 1

            events.append({
                "name": "note_sustain", 
                "time": 0, 
                "pitch": pitch, 
                "velocity": velocity
            })

        elif seg_start <= note.start <= seg_end and seg_start <= note.end <= seg_end:

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx, pitch] += 1
            offset_roll[offset_idx, pitch] += 1
            frame_roll[onset_idx : offset_idx + 1, pitch] += 1

            events.append({
                "name": "note_on",
                "time": onset_time, 
                "pitch": pitch, 
                "velocity": velocity
            })
            events.append({
                "name": "note_off",
                "time": offset_time, 
                "pitch": pitch, 
            })

        elif seg_start <= note.start <= seg_end and seg_end < note.end < math.inf:

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx, pitch] += 1
            frame_roll[onset_idx : , pitch] += 1

            events.append({
                "name": "note_on",
                "time": onset_time, 
                "pitch": pitch, 
                "velocity": velocity
            })

    events.sort(key=lambda event: event["time"])

    data = {
        "frame_roll": frame_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "velocity_roll": offset_roll,
        "events": events,
    }

    return data


def pedals_to_targets(pedals, segment_frames, segment_start, segment_end, fps):

    #
    seg_start = segment_start
    seg_end = segment_end
    pitches_num = 128

    # Covert pedals information to words.
    frame_roll = np.zeros(segment_frames)
    onset_roll = np.zeros(segment_frames)
    offset_roll = np.zeros(segment_frames)

    events = []

    for pedal in pedals:

        onset_time = pedal.start - seg_start
        offset_time = pedal.end - seg_start
        
        if 0 <= pedal.start < seg_start and seg_start <= pedal.end < seg_end:

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx] += 1
            frame_roll[0 : offset_idx + 1] += 1

            events.append({
                "name": "pedal_sustain", 
                "time": 0, 
            })
            events.append({
                "name": "pedal_off",
                "time": offset_time, 
            })

        if 0 <= pedal.start < seg_start and seg_end <= pedal.end < math.inf:

            frame_roll += 1

            events.append({
                "name": "pedal_sustain", 
                "time": 0, 
            })

        elif seg_start <= pedal.start <= seg_end and seg_start <= pedal.end <= seg_end:

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx] += 1
            offset_roll[offset_idx] += 1
            frame_roll[onset_idx : offset_idx + 1] += 1

            events.append({
                "name": "pedal_on",
                "time": onset_time, 
            })
            events.append({
                "name": "pedal_off",
                "time": offset_time,
            })

        elif seg_start <= pedal.start <= seg_end and seg_end < pedal.end < math.inf:

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx] += 1
            frame_roll[onset_idx :] += 1

            events.append({
                "name": "pedal_on",
                "time": onset_time, 
            })

    events.sort(key=lambda event: event["time"])

    data = {
        "frame_roll": frame_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "events": events,
    }

    return data


def add():

    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    notes, pedals = read_single_track_midi(midi_path)
    
    write_notes_to_midi(notes, "_zz.mid")

    segment_frames = 1001
    seg_start = 25.
    seg_end = 35.
    fps = 100

    t1 = time.time()
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

    frame_roll = note_data["frame_roll"]
    onset_roll = note_data["onset_roll"]
    offset_roll = note_data["offset_roll"]
    velocity_roll = note_data["velocity_roll"]

    ped_frame_roll = pedal_data["frame_roll"]
    ped_onset_roll = pedal_data["onset_roll"]
    ped_offset_roll = pedal_data["offset_roll"]

    all_events = note_data["events"] + pedal_data["events"]
    all_events.sort(key=lambda event: event["time"])

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add()