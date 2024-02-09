import torch

from pathlib import Path
from model.lstm import AirModel
from data.dataset import Dataset
from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

if __name__ == '__main__':
    """
    Neural network visualizer tested with RNN MNIST digit classifier. 
    """
    current = Path.cwd()
    path = current.joinpath('state/lstm.pth')
    state = torch.load(path)

    model = AirModel()
    model.load_state_dict(state)
    model.eval()

    dataset = Dataset()
    model_input = []
    i = 0

    for x_batch, _ in dataset.train_loader: 
        info = {
            'file' : str(i),
            'data' : x_batch
        }

        model_input.append(info)

        i += 1

        if i > 5:
            break

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()
    