from __future__ import annotations

import inquirer
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from pathlib import Path

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
    """

    def __init__(self, visual: Visual = None, folder: str = None):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.  
        """
        self.visual = visual
        self.folder = folder


    @abstractmethod
    def visualize(self, data: Any):
        pass


    def save(
        self,
        file: str,
        model: str,
        layer: str,
        layer_num: str
    ):
        """Saves figure to folder.

        Args:
            file: name for input into model.
            model: model the figure was taken from.
            layer: name of layer the figure was created for.
            layer_num: number for layers with multiple activation maps.
        """
        file = "file_" + file[:file.find('.')]

        path = Path.cwd().joinpath('Visualization')
        path = path.joinpath(self.folder).joinpath(file)
        path = path.joinpath(model).joinpath(layer)
        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath(layer_num + ".jpg")

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
