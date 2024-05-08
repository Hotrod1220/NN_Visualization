import torch

from pathlib import Path

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from model.rnn_speech import RNN
from data.rnn_dataset import Dataset
# from visual_methods.heatmap import Heatmap
# from visual_methods.heatmaps import Heatmaps
from visual_methods.heatmap_mining import HeatmapMining
from visualization import Visualization
from data_mining.series import Series


def main():
    """
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    """
    dataset = Dataset()

    n_hidden = 128
    n_letters = dataset.n_letters
    n_categories = dataset.n_categories

    current = Path.cwd()
    path = current.joinpath('state/rnn_text.pth')
    state = torch.load(path)

    rnn = RNN(n_letters, n_hidden, n_categories)
    rnn.load_state_dict(state)
    rnn.eval()

    # category, line, category_tensor, line_tensor =  dataset.randomTrainingExample()

    # hidden = rnn.initHidden()
    model_input = []

    for j in range(5):
        category, line, category_tensor, line_tensor =  dataset.randomTrainingExample()

        hidden = rnn.initHidden()
        
        for i in range(line_tensor.size()[0]):
            info = {
                'name' : str(line[i]),
                'data' : (line_tensor[i], hidden),
                'labels' : dataset.all_categories
            }

            model_input.append(info)

    mining = Series()
    visual = HeatmapMining(predictions=True, mining=mining)
    visualization = Visualization(rnn, model_input, visual)
    visualization.visualize()


if __name__ == '__main__':
    main()