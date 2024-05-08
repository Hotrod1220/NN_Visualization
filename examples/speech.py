import torch
import torchaudio
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import wavfile

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from model.speech import M5
from data.speech_dataset import SubsetSC
from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

def main():
    """
    Neural network visualizer tested with speech classifier. 
    https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
    """
    current = Path.cwd()
    path = current.joinpath('state/m5.pth')
    state = torch.load(path)

    test_set = SubsetSC("validation")

    waveform, sample_rate, _, _, _ = test_set[0]
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    new_sample = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq = sample_rate,
        new_freq = new_sample
    )
    
    transformed = transform(waveform)
    model = M5(
        n_input = transformed.shape[0],
        n_output = len(labels)
    )

    model.load_state_dict(state)
    model.eval()

    # video(test_set, transform, new_sample, labels, model)
    phonemes(labels, model, new_sample)


def phonemes(labels, model, new_sample):
    """Passing english phonemes into the model to see how the activations react."""
    model_input = []
    path = Path.cwd().joinpath("data/english-phonemes")

    for p in path.rglob("*"):
        f_path = path.joinpath(p)
        if f_path.is_file() and f_path.suffix == '.wav':
            sample_rate, waveform = wavfile.read(str(f_path))

            transform = torchaudio.transforms.Resample(
                orig_freq = sample_rate,
                new_freq = new_sample
            )

            waveform = waveform.flatten()
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            waveform = transform(waveform)
            # plot_waveform(waveform, new_sample, f_path.stem) 
            
            start = waveform.shape[1] // 2 - 4000
            end = waveform.shape[1] // 2 + 4000
            waveform = waveform = waveform[0][start : end]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            # waveform = waveform.unsqueeze(0)

            file = f_path.stem + "_split"
            info = {
                'name' : file,
                'data' : waveform,
                'labels' : labels
            }

            model_input.append(info)

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()


def video(test_set, transform, new_sample, labels, model):
    """Slicing english words in increasing 10ms intervals into the 
    model and creates a video.
    """
    for i in range(0, 10000, 1000):
        model_input = []
        waveform, _, label, _, *_ = test_set[i]
        waveform = transform(waveform)

        for j in range(80, 8000, 80):
            file = f"{label}_{j // 8}ms"

            new_waveform = slice_waveform(waveform, new_sample, 0, j)
            
            # path = current.joinpath(f"data/sound/{file}.wav")
            # torchaudio.save(
            #     str(path),
            #     new_waveform,
            #     new_sample
            # )
            # plot_waveform(new_waveform, new_sample, file)
            
            new_waveform = new_waveform.unsqueeze(0)

            info = {
                'name' : file,
                'data' : new_waveform,
                'labels' : labels
            }

            model_input.append(info)

        visual = Heatmaps(video = True)
        visualization = Visualization(model, model_input, visual)
        visualization.visualize()


def slice_waveform(waveform, sample_rate, start, end):
    waveform = waveform[0][start : end]
    pad_end = int(sample_rate - end)
    pad = (1 * start, 1 * pad_end)
    waveform = torch.nn.functional.pad(waveform, pad, value=0)
    waveform = waveform.unsqueeze(0)
    return waveform


def plot_waveform(waveform, sample_rate, name):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

    path = Path.cwd().joinpath(f"data/sound/{name}.png")
    plt.savefig(path)
    

if __name__ == "__main__":
    main()