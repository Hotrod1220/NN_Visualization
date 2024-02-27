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

    train_set = SubsetSC("testing")

    waveform, sample_rate, _, _, _ = train_set[0]
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

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

    for i in range(0, 10000, 1000): 
        waveform, _, label, _, *_ = train_set[i]
        waveform = transform(waveform)
        waveform = waveform.unsqueeze(0)

        info = {
            'file' : label,
            'data' : waveform
        }

        model_input.append(info)

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()

if __name__ == "__main__":
    main()