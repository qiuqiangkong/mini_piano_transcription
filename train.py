import torch
import torch.nn.functional as F
import time
import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
from data.maestro import MaestroDataset
from data.collate import collate_fn
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    epochs = 2000
    checkpoints_dir = Path("./checkpoints", model_name)
    debug = False
    
    root = "/home/qiuqiangkong/datasets/maestro-v2.0.0"

    # Dataset
    dataset = MaestroDataset(
        root=root,
        split="train",
        # segment_seconds=4.,
        segment_seconds=10.,
    )

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=8, 
        collate_fn=collate_fn,
        num_workers=8, 
        # num_workers=0, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name)
    model.to(device)

    # checkpoint_path = Path("checkpoints", model_name, "latest.pth")
    # model.load_state_dict(torch.load(checkpoint_path))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(1, epochs):
        
        for data in tqdm(dataloader):

            audio = data["audio"].to(device)
            onsets_roll = data["onsets_roll"].to(device)

            # soundfile.write(file="_zz.wav", data=audio.cpu().numpy()[0], samplerate=16000)

            # fig, axs = plt.subplots(2,1, sharex=True)
            # axs[0].matshow(onsets_roll.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            # axs[1].matshow(data["frames_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig("_zz.pdf")
            

            # Play the audio.
            if debug:
                play_audio(mixture, target)

            optimizer.zero_grad()

            model.train()
            output_dict = model(audio=audio) 

            # fig, axs = plt.subplots(2,1, sharex=True)
            # axs[0].matshow(onsets_roll.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            # axs[1].matshow(output_dict["onset_roll"].data.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig("_zz.pdf")

            # from IPython import embed; embed(using=False); os._exit(0)

            loss = bce_loss(output_dict["onset_roll"], onsets_roll)
            loss.backward()

            optimizer.step()

        print(loss)

        # Save model
        if epoch % 20 == 0:
            checkpoint_path = Path(checkpoints_dir, "epoch={}.pth".format(epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))


def get_model(model_name):
    if model_name == "CRnn":
        return CRnn()
    elif model_name == "CRnn2":
        from models.crnn2 import CRnn2
        return CRnn2()
    elif model_name == "CRnn3":
        from models.crnn3 import CRnn3
        return CRnn3()
    else:
        raise NotImplementedError


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRnn")
    args = parser.parse_args()

    train(args)