import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

from model.rnn import RNN
from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

if __name__ == '__main__':
    """
    Neural network visualizer tested with RNN MNIST digit classifier. 
    """
    current = Path.cwd()
    path = current.joinpath('state/rnn.pth')
    state = torch.load(path)

    model = RNN()
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

        info = {
            'file' : name,
            'data' : image,
            'digit' : i
        }

        model_input.append(info)

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()
    