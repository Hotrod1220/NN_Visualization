from __future__ import annotations

import os
import imageio
import torch
import inquirer
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from pathlib import Path
from natsort import natsorted 

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class Visual(ABC):
    """Abstract class for different methods of visualizing neural networks.

    This class employs the decorator design pattern, providing flexibility
    in selecting and combining different visualization methods.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        video: Whether video is wanted for some visualizations.
    """

    def __init__(
        self,
        visual: Visual = None,
        folder: str = None,
        video: bool = False
    ):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.
            video: Whether video is wanted for some visualizations.  
        """
        self.visual = visual
        self.folder = folder
        self.video = video


    @abstractmethod
    def visualize(self, data: Any):
        pass


    def save(
        self,
        file: str,
        model: str,
        layer: str = None,
        layer_num: str = None
    ):
        """Saves figure to folder.

        Args:
            file: name for input into model.
            model: model the figure was taken from.
            layer: name of layer the figure was created for.
            layer_num: number for layers with multiple activation maps.
        """
        ext = file.find('.')
        ext = file[:ext] if ext != -1 else file
        if ext != '':
            file = "file_" + ext
        else:
            file = "file_" + file

        path = Path.cwd().joinpath('Visualization')
        path = path.joinpath(self.folder).joinpath(file)

        if layer is None and layer_num is None:
            path.mkdir(parents=True, exist_ok=True)
            path = path.joinpath(model + ".png")
        else: 
            path = path.joinpath(model).joinpath(layer)
            path.mkdir(parents=True, exist_ok=True)
            path = path.joinpath(layer_num + ".png")

        plt.savefig(path)
        plt.clf()


    def select_layers(self, data: list[dict]) -> dict[str, list[str]]:
        """Prompts the user to select the layers they would like to visualize.

        Args:
            data: model and layer data

        Returns:
            Which models and which layers for each model to visualize.
        """
        visual_layers = {}
        print("\n")

        model_options = [entry for entry in data['activations']]

        model_question = [inquirer.Checkbox(
            'models',
            message = "Which models would you like to visualize? "
                "Press Enter for all",
            choices = model_options
        )]

        models = inquirer.prompt(model_question)

        if len(models['models']) == 0:
            models = model_options
        else:
            models = models['models']

        for model in models:
            layer_options = [
                name 
                for name, value in data['activations'][model].items()
            ]

            layer_question = [inquirer.Checkbox(
                'layers',
                message = f"Model: {model}, Which layers would you like to "
                    "visualize? Press Enter for all",
                choices = layer_options
            )]

            layers = inquirer.prompt(layer_question)

            if len(layers['layers']) == 0:
                layers = layer_options
            else:
                layers = layers['layers']

            visual_layers[model] = layers

        return visual_layers


    def activations_2D(self, array: np.ndarray | torch.Tensor) -> list[np.ndarray]:
        """Decomposes larger dimension arrays into all its 2D arrays.

        Args:
            array: array to get 2D arrays from.

        Returns:
            list of 2D numpy arrays contained in larger dimension array.
        """
        if isinstance(array, torch.Tensor):
            array = array.squeeze()
            array = array.detach().cpu().numpy()

        activations = []

        if len(array.shape) <= 2:
            return array
            
        for i in range(array.shape[0]):
            activations.append(self.activations_2D(array[i]))
        
        return activations
    

    def save_videos(self, data: dict[str, dict[str, torch.Tensor]]) -> None:
        """Visualizes selected neural network layers with heatmaps.

        Args:
            data: File name, model / layer data, and other attributes.
        """
        if self.video:
            path = Path.cwd().joinpath("Visualization")
            path = path.joinpath(self.folder)
            path = path.joinpath(f"file_{data['file']}")

            models = list(data['activations'].keys())
            models = [
                model[:model.find('_')]
                for model in models
                if model.find('_') != -1
            ]

            dup = set()
            models = [
                model 
                for model in models
                if model in dup or dup.add(model)
            ]  

            for name in dup:
                files = []
                layers = {}
                for file in os.listdir(path):
                    if file.startswith(name):
                        file_path = path.joinpath(file)
                        if file_path.is_file():
                            files.append(file_path)
                        else:
                            for p in file_path.rglob("*"):
                                f_path = file_path.joinpath(p)
                                if f_path.is_file() and f_path.suffix == '.png':
                                    files.append(f_path)

                for file in files:
                    if file.parts[-2] not in layers:
                        layers[file.parts[-2]] = [file]
                    else:
                        layers[file.parts[-2]].append(file)

                for folder, files in layers.items():
                    if len(layers) > 1:
                        new_path = path.joinpath(name)
                    else:
                        new_path = path
                        folder = name
                    if len(files) > 1:
                        self.create_video(folder, files, new_path)
        
            folders = list(os.walk(path))[1:]

            for folder in folders:
                if not folder[1] and not folder[2]:
                    os.rmdir(folder[0])


    def create_video(self, name: str, files: list, path: Path) -> None:
        """Creates a video from list of image files.
         
        Args:
            name: video name.
            files: list of image file paths.
            path: folder for saving video.
        """
        files = natsorted(files)
        
        path = path.joinpath(f'{name}.mp4')
        if path.is_file():
            path.unlink()

        video_name = str(path)
        writer = imageio.get_writer(video_name, fps=5)

        for im in files:
            writer.append_data(imageio.imread(im))
        writer.close()

        for file in files:
            file.unlink()
