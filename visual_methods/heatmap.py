from __future__ import annotations

import copy
import seaborn as sns

from visualize import Visual

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np


class Heatmap(Visual):
    """Visualizes neural network activation layers with a heatmap.

    Generates and saves a full heatmap for each feature map in each layer.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        layers: Layers selected for visualization.
    """

    def __init__(
        self,
        visual: Visual  =  None,
        folder: str  =  "Heatmap",
    ):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.
        """
        super().__init__(visual, folder)
        self._layers  =  None


    def visualize(self, data: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        """Visualizes selected neural network layers with individual heatmaps.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
        """
        visual_data  =  copy.deepcopy(data)

        if self._layers is None:
            self._layers  =  self.select_layers(data)

        for model, layer in self._layers.items():
            print(f"\nVisualizing model: {model}")

            for entry in data:
                print(f"\nVisualizing: {entry['name']}")
                for name in layer:
                    print(f"Visualizing layer: {name}")

                    activation  =  entry['activations'][model][name]
                    activation  =  activation.squeeze()
                    activation  =  activation.detach().cpu().numpy()

                    activations  =  self.activations_2D(activation)

                    if not isinstance(activations, list):
                        activations  =  [activations]

                    layer_num  =  1
                    for values in activations:
                        if name == "Output" and 'labels' in entry:
                            self.plot_heatmap(values, entry['labels'])
                        else:
                            self.plot_heatmap(values)

                        self.save(
                            entry['name'],
                            model,
                            name,
                            f"layer_{layer_num}"
                        )
                        
                        layer_num += 1

        if self.visual is not None:
            self.visual.visualize(visual_data)


    def plot_heatmap(
        self,
        data: np.array,
        labels: list[str | int | float]  =  None
    ) -> None:
        """Plots the data into a heatmap.

        Args:
            data: 1D or 2D np.array used for heatmap visualization.
            labels: labels used for the output layer.
        """
        if labels is not None:
            if len(data.shape) == 1:
                sns.heatmap(
                    [data],
                    xticklabels = labels,
                    cbar_kws = {"orientation": "horizontal"}
                )
            else:
                sns.heatmap(data, xticklabels = labels)
        else:
            if len(data.shape) == 1:
                sns.heatmap(
                    [data],
                    cbar_kws = {"orientation": "horizontal"}
                )
            else:
                sns.heatmap(data)
