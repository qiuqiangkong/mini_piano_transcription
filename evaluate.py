import argparse
import mir_eval
import pretty_midi
import numpy as np


def evaluate(args):
    
    ref_midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi"
    
    est_midi_path = "./_zz.mid"

    ref_intervals, ref_pitches = parse_midi(ref_midi_path)
    est_intervals, est_pitches = parse_midi(est_midi_path)

    note_precision, note_recall, note_f1, _ = \
    mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals, 
        ref_pitches=ref_pitches, 
        est_intervals=est_intervals, 
        est_pitches=est_pitches, 
        onset_tolerance=0.05, 
        offset_ratio=None,)

    print("F1: {:.3f}".format(note_f1))

    from IPython import embed; embed(using=False); os._exit(0)


def parse_midi(midi_path):

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    notes = midi_data.instruments[0].notes

    intervals = []
    pitches = []

    for note in notes:
        intervals.append([note.start, note.end])
        pitches.append(note.pitch)

    return np.array(intervals), np.array(pitches)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    evaluate(args) 
