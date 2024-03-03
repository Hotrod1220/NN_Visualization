import torch
import torchaudio
import matplotlib.pyplot as plt

from pathlib import Path

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
    """
    current = Path.cwd()
    path = current.joinpath('state/m5.pth')
    state = torch.load(path)

    test_set = SubsetSC("validation")

    waveform, sample_rate, _, _, _ = test_set[0]
    # labels = sorted(list(set(datapoint[2] for datapoint in test_set)))
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

    model_input = []

    # for i in range(0, 10000, 1000):
    for i in range(1):
        waveform, _, label, _, *_ = test_set[i]
        waveform = transform(waveform)

        file = f"{label}"
        # waveform = slice_waveform(waveform, new_sample, 5000, 8000)
        
        path = current.joinpath(f"data/sound/{file}.wav")

        torchaudio.save(
            str(path),
            waveform,
            new_sample
        )

        plot_waveform(waveform, new_sample, file)
        waveform = waveform.unsqueeze(0)

        info = {
            'file' : file,
            'data' : waveform,
            'labels' : labels
        }

        model_input.append(info)

    visual = Heatmaps()
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