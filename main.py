import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import Model
from heatmap import Heatmap
from visualization import Visualization

if __name__ == '__main__':
    """
    Neural network visualizer tested with MNIST digit classifier. 
    """
    current = Path.cwd()
    path = current.joinpath('state/model.pth')
    state = torch.load(path)

    model = Model()
    model.load_state_dict(state)
    model.eval()

    current = current.joinpath('images')
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
            'digit' : i
        }

        model_input.append(info)

    visual = Heatmap()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()
    