from __future__ import annotations

import seaborn as sns
import numpy as np

from visualize import Visual


class Heatmap(Visual):
    """Visualizes neural network activation layers with a heatmap.

    Generates and saves a full heatmap for each layer.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        layers: Layers selected for visualization.
    """

    def __init__(self, visual: Visual = None, folder: str = "Heatmap"):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.  
        """
        super().__init__(visual, folder)
        self._layers = None


    def visualize(self, data: list[dict]) -> None:
        """Visualizes selected neural network layers with heatmaps.

        Args:
            data: file name, model and layer data, and other attributes.
        """
        if self._layers is None:
            self._layers = self.select_layers(data)

        for model, layer in self._layers.items():
            print(f"\nVisualizing model: {model}")

            for name in layer:
                print(f"Visualizing layer: {name}")

                activation = data['activations'][model][name]
                activation = activation.squeeze()
                activation = activation.detach().cpu().numpy()

                activations = self.activations_2D(activation)

                layer_num = 1
                for values in activations:
                    self.plot_heatmap(values)
                    self.save(
                        data['file'],
                        model,
                        name,
                        f"layer_{layer_num}"
                    )
                    
                    layer_num += 1 

        if self.visual is not None:
            self.visual.visualize(data)


    def plot_heatmap(self, data: np.array) -> None:
        """Plots the data into a heatmap.

        Args:
            data: 1D or 2D np.array used for heatmap visualization.
        """
        if len(data.shape) == 1:
            sns.heatmap(
                [data],
                square = True,
                cbar_kws={"orientation": "horizontal"}
            )
        else:
            sns.heatmap(data, square = True)
        

    def activations_2D(self, array: np.array) -> list[np.array]:
        """Decomposes larger dimension arrays into all its 2D arrays.

        Args:
            array: array to get 2D arrays from.

        Returns:
            list of 2D numpy arrays contained in larger dimension array.
        """
        activations = []

        if len(array.shape) == 2:
            return array
        
        if len(array.shape) == 1:
            return [array]
            
        for i in range(array.shape[0]):
            activations.append(self.activations_2D(array[i]))
        
        return activations
