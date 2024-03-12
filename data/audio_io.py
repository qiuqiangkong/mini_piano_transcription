import numpy as np
import random
import librosa
import torch
import torchaudio


class AudioIO:
    def __init__(self):
        pass

    def random_start_time(self, duration):

        if self._segment_seconds is None:
            # Full audio.
            seg_start_time = 0
            seg_seconds = duration

        else:
            # Random segment of an audio.
            if duration < self._segment_seconds:
                seg_start_time = 0
            else:
                seg_start_time = random.uniform(0, duration - self._segment_seconds)
                
            seg_seconds = self._segment_seconds

        return seg_start_time, seg_seconds

    def load_audio(self, audio_path, segment_start_time, segment_seconds):

        orig_sr = librosa.get_samplerate(audio_path)

        orig_seg_start_sample = round(segment_start_time * orig_sr)
        orig_seg_samples = round(segment_seconds * orig_sr)

        audio, fs = torchaudio.load(
            audio_path, 
            frame_offset=orig_seg_start_sample, 
            num_frames=orig_seg_samples
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

        new_seg_samples = round(segment_seconds * self.sample_rate)
        
        audio = librosa.util.fix_length(
            data=np.array(audio), 
            size=new_seg_samples, 
            axis=0
        )

        return audio