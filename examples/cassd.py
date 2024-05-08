import torch
import numpy as np

from pathlib import Path
from PIL import Image

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

def main():
    """
    Neural network visualizer tested with multiple MNIST 
    digit localizer and classifier. 
    """
    model = torch.load('state/cassd.pth', map_location='cpu')
    model.eval()

    path = Path.cwd().joinpath('data/images/image.png')

    with Image.open(path).convert('L') as image:
        image = np.asarray(image).astype('uint8')

    tensor = (
        torch
        .tensor(image)
        .to(dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to('cpu')
    )

    info = {
        'name' : 'image.png',
        'data' : tensor
    }

    visual = Heatmaps()
    visualization = Visualization(model, [info], visual)
    visualization.visualize()

if __name__ == '__main__':
    main()