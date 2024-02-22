import torch
import time
import pretty_midi
import copy

from data.maestro import Maestro
from data.collate import collate_fn
from train import Sampler


def add():
    root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"

    dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=10.,
    )

    sampler = Sampler(dataset_size=len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    for _ in range(100):
        # t1 = time.time()
        data = dataset[0]
        # print(time.time() - t1)

    # for data in dataloader:
    #     from IPython import embed; embed(using=False); os._exit(0)


def add2():

    # midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes

    tmp = copy.deepcopy(notes)
    tmp2 = copy.deepcopy(control_changes)

    onset = None
    offset = None
    pairs = []

    t1 = time.time()
    
    notes.sort(key=lambda note: note.start)
    control_changes.sort(key=lambda cc: cc.time)

    print("a1", time.time() - t1)

    #
    t1 = time.time()
    for cc in control_changes:

        if cc.number == 64:

            if cc.value >= 64 and onset is None:
                onset = cc.time

            elif cc.value < 64 and onset is not None:
                offset = cc.time
                pair = (onset, offset)
                pairs.append(pair)

                onset = None
                offset = None

    if onset is not None and offset is None:
        pair = (onset, control_changes[-1].time)
        pairs.append(pair)

    
    print("a2", time.time() - t1)
    # from IPython import embed; embed(using=False); os._exit(0)

    t1 = time.time()

    notes.sort(key=lambda note: note.end)

    tmp3 = copy.deepcopy(pairs)

    pedals = pairs
    # new_notes = []

    new_notes = {pitch: [] for pitch in range(128)}

    while len(pedals) > 0 and len(notes) > 0:

        ped_on, ped_off = pedals[0]

        while notes:

            note = notes[0]

            if note.end < ped_on:
                new_notes[note.pitch].append(note)
                notes.pop(0)

            elif ped_on <= note.end < ped_off:
                new_note = pretty_midi.Note(
                    pitch=note.pitch, 
                    start=note.start, 
                    end=ped_off, 
                    velocity=note.velocity
                )
                if len(new_notes[note.pitch]) > 0:
                    if new_notes[note.pitch][-1].end > new_note.start:
                        new_notes[note.pitch][-1].end = new_note.start
                new_notes[note.pitch].append(new_note)
                notes.pop(0)

            elif ped_off <= note.end:
                pedals.pop(0)
                break

            else:
                raise NotImplementedError            

    from IPython import embed; embed(using=False); os._exit(0)

    new_notes.extend(notes)
    # while True:
    #     new_notes.append(note)
    #     notes.pop(0)

    print("a3", time.time() - t1)

    
    new_notes.sort(key=lambda note: note.start)

    print("a3", time.time() - t1)
    
    new_midi_data = pretty_midi.PrettyMIDI()
    new_track = pretty_midi.Instrument(program=0)
    for new_note in new_notes:
        new_track.notes.append(new_note)
    new_midi_data.instruments.append(new_track)
    new_midi_data.write('_zz.mid')

    from IPython import embed; embed(using=False); os._exit(0)

    # 

def add3():

    a1 = [1,2,3,4,5]

    for i in a1:
        print(i)
        a1.pop(0)
        a1.pop(0)
        # a1.pop()


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

    


def add4():

    
    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes

    t1 = time.time()
    pedals = get_pedals(control_changes)

    notes = extend_offset_by_pedal(notes, pedals)
    print(time.time() - t1)

    #    
    new_midi_data = pretty_midi.PrettyMIDI()
    new_track = pretty_midi.Instrument(program=0)
    for note in notes:
        new_track.notes.append(note)
    new_midi_data.instruments.append(new_track)
    new_midi_data.write('_zz.mid')

    #
    segment_frames = 1001
    pitches_num = 128
    seg_start = 25.
    seg_end = 35.
    fps = 100

    
    

    # Covert notes information to words.
    frame_roll = np.zeros((segment_frames, pitches_num))
    onset_roll = np.zeros((segment_frames, pitches_num))
    offset_roll = np.zeros((segment_frames, pitches_num))
    velocity_roll = np.zeros((segment_frames, pitches_num))

    events = []

    for note in range(len(notes)):
        # if segment_start <= notes[i].start < segment_end:
        #     active_notes.append(notes[i])
        if note.start < seg_start and seg_start <= note.end < seg_end:

            offset_time = note.end - seg_start
            pitch = note.pitch

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx, pitch] += 1
            frame_roll[0 : offset_idx + 1, pitch] += 1

            event = {"time": offset_time, "pitch": pitch, "velocity": 0}
            events.append(event)
            
        if note.start < seg_start and seg_end <= note.end:

            pitch = note.pitch
            frame_roll[:, pitch] += 1

        elif seg_start <= note.start < seg_end and seg_start <= note.end <= seg_end:

            onset_time = note.start - seg_start
            offset_time = note.end - seg_start
            pitch = note.pitch
            velocity = note.velocity

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx, pitch] += 1
            offset_roll[offset_idx, pitch] += 1
            frame_roll[onset_idx : offset_idx + 1, pitch] += 1

            events = append({"time": onset_time, "pitch": pitch, "velocity": velocity})
            events = append({"time": offset_time, "pitch": pitch, "velocity": 0})

        elif seg_start <= note.start < seg_end and seg_end <= note.end:

            onset_time = note.start - seg_start
            pitch = note.pitch
            velocity = note.velocity

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx, pitch] += 1
            frame_roll[onset_idx : , pitch] += 1

            events = append({"time": onset_time, "pitch": pitch, "velocity": velocity})






    words = ["<sos>"]


    for note in active_notes:

        onset_time = note.start - segment_start
        offset_time = note.end - segment_end
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


if __name__ == '__main__':
    add4()