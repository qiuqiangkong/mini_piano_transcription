import torch
import time
import pickle
import librosa
import numpy as np
import soundfile
import pretty_midi
from pathlib import Path
import torch.optim as optim
from data.maestro import MaestroDataset
from data.collate import collate_fn
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import matplotlib.pyplot as plt


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000

    # Load checkpoint
    checkpoint_path = Path("checkpoints", model_name, "latest.pth")

    model = CRnn()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Load audio. Change this path to your favorite song.
    # root = "/home/qiuqiangkong/datasets/musdb18hq/test"
    # mixture_path = Path(root, "Al James - Schoolboy Facination", "mixture.wav") 
    audio_path = "/home/qiuqiangkong/datasets/maestro/maestro-v2.0.0/2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav"

    audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
    # (channels_num, audio_samples)

    audio_samples = audio.shape[-1]
    bgn = 0
    segment_samples = int(segment_seconds * sample_rate)

    onset_rolls = []

    # Do separation
    while bgn < audio_samples:
        
        print("Processing: {:.1f} s".format(bgn / sample_rate))

        # Cut segments
        segment = audio[bgn : bgn + segment_samples]
        segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)
        segment = torch.Tensor(segment).to(device)

        # Separate a segment
        with torch.no_grad():
            model.eval()
            output_dict = model(audio=segment[None, :])
            onset_roll = output_dict["onset_roll"].cpu().numpy()[0]
            onset_rolls.append(onset_roll[0:-1, :])
            # sep_wavs.append(sep_wav.cpu().numpy())

        bgn += segment_samples

        # soundfile.write(file="_zz.wav", data=segment.cpu().numpy(), samplerate=sample_rate)

        # plt.matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

    onset_rolls = np.concatenate(onset_rolls, axis=0)
    pickle.dump(onset_rolls, open("_zz.pkl", "wb"))
    
    post_process()

def post_process():
    onset_roll = pickle.load(open("_zz.pkl", "rb"))

    # import torch
    # a1 = torch.Tensor(onset_roll[None, None, :, :])
    # w = torch.Tensor(np.zeros((1, 1, 5, 5)))
    # y = torch.nn.functional.conv2d(input=a1, weight=w, padding=2)
    # y = y.cpu().numpy()[0, 0]
    # plt.matshow(onset_roll[0:1000].T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    frames_num, pitches_num = onset_roll.shape

    notes = []

    for k in range(pitches_num):
        for n in range(frames_num):
            if onset_roll[n, k] > 0.5:
                onset = n / 100
                offset = onset + 0.2
                note = {
                    "onset": onset,
                    "offset": offset,
                    "pitch": k
                }
                notes.append(note)
                
    # Write to MIDI
    new_midi_data = pretty_midi.PrettyMIDI()
    new_track = pretty_midi.Instrument(program=0)
    for note in notes:
        midi_note = pretty_midi.Note(pitch=note["pitch"], start=note["onset"], end=note["offset"], velocity=100)
        new_track.notes.append(midi_note)
    new_midi_data.instruments.append(new_track)
    new_midi_data.write('_zz.mid')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRnn")
    args = parser.parse_args()

    inference(args) 
