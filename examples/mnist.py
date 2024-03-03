import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from model.model import Model
from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

def main():
    """
    Neural network visualizer tested with MNIST digit classifier. 
    """
    current = Path.cwd()
    path = current.joinpath('state/model.pth')
    state = torch.load(path)

    model = Model()
    model.load_state_dict(state)
    model.eval()

    current = current.joinpath('data/images')
    model_input = []

    for i in range(0, 10): 
        name = f"{i}.jpg"
        image_path = current.joinpath(name)
        image = Image.open(image_path)

        transform = transforms.Compose([transforms.PILToTensor()])
        image = transform(image)
        image = image.float()
        image /= 255
        image = image.unsqueeze(0)

        info = {
            'file' : name,
            'data' : image,
            'digit' : i,
            'labels' : [j for j in range(10)]
        }

        model_input.append(info)

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()

if __name__ == "__main__":
    main()