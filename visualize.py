from __future__ import annotations

import torch
import inquirer
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Visual(ABC):
    """Abstract class for different methods of visualizing neural networks.

    This class employs the decorator design pattern, providing flexibility
    in selecting and combining different visualization methods.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
    """

    def __init__(
        self,
        visual: Visual = None,
        folder: str = None,
    ):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.
        """
        self.visual = visual
        self.folder = folder


    @abstractmethod
    def visualize(self, data: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        pass


    def save(
        self,
        file: str,
        model: str,
        layer: str = None,
        layer_num: str = None
    ) -> None:
        """Saves figure to folder.

        Args:
            file: name for input into model.
            model: model the figure was taken from.
            layer: name of layer the figure was created for.
            layer_num: number for layers with multiple activation maps.
        """
        ext = file.find('.')
        ext = file[:ext] if ext != -1 else file
        file = ext if ext != '' else file

        path = Path.cwd().joinpath('Visualization')
        path = path.joinpath(self.folder).joinpath(model)

        if layer is None and layer_num is None:
            path.mkdir(parents=True, exist_ok=True)
            path = path.joinpath(file + ".png")
        else:
            file = "file_" + file
            path = path.joinpath(file).joinpath(layer)
            path.mkdir(parents=True, exist_ok=True)
            path = path.joinpath(layer_num + ".png")

        plt.savefig(path)
        plt.close()


    def select_layers(
            self,
            data: list[dict[str, dict[str, torch.Tensor]]]
        ) -> dict[str, list[str]]:
        """Prompts the user to select the layers they would like to visualize.

        Args:
            data: model and layer data

        Returns:
            Which models and which layers for each model to visualize.
        """
        visual_layers = {}
        print("\n")

        data = data[0]
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
