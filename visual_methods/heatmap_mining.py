# from __future__ import annotations

import os
import sys
import copy
import torch
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use("Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from visualize import Visual

# from typing import TYPE_CHECKING

current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from data_mining.series import Series
from data_mining.mining import DataMining

# if TYPE_CHECKING:
    

class HeatmapMining(Visual):
    """Uses data mining and heatmaps to visualize neural network activation layers.

    Extracts useful data for visualizing activations over all inputs, top predictions
    is an optional addition. Useful for visualizing RNNs and LSTMs.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        predictions: If top predictions should be visualized.
        mining: Data mining methods. 
    """

    def __init__(
        self,
        visual: Visual = None,
        folder: str = "Heatmap_Series",
        predictions: bool = False,
        mining: list[DataMining] | DataMining = None
    ):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.
            predictions: If top predictions should be visualized.
            mining: Data mining methods. 
        """
        super().__init__(visual, folder)
        self.predictions = predictions
        self.mining = mining


    @property
    def mining(self):
        """Data mining methods to be used for visualizing neural networks.

        Raises:
            Exception if value in list is not of type DataMining.
        """
        return self._mining
    

    @mining.setter
    def mining(self, obj: list[DataMining] | DataMining | None):
        if obj is not None:
            if not isinstance(obj, list):
                obj = [obj]

            for mining in obj:
                if not isinstance(mining, DataMining):
                    Exception("Mining methods must be of type DataMining.")
        self._mining = obj


    def visualize(self, data: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        """Visualizes neural network layers of all inputs with heatmaps.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
        """
        visual_data = copy.deepcopy(data)

        if self.mining is None:
            self.mining = [Series()]
        
        for mine in self.mining:
            mine.data = data
            ext_data = mine.extract()

        # data_line = 100
        # data = self.remove_hidden(data)
        # fig = self.init_plot(data, data_line)
        # averages = self.averages(data)
        # plot_data = self.add_predictions(data, averages)
        # self.plot(averages)

        if self.visual is not None:
            self.visual.visualize(visual_data)


    def init_plot(
        self,
        data: list[dict[str, dict[str, torch.Tensor]]],
        data_line: int
    ) -> plt.Figure:
        """Initializes the plot for a model. Height depends on amount of data present.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
            data_line: Amount of data entries per line.

        Returns:
            Figure to plot data on.
        """
        height = len(data) / data_line + 20 if len(data) > data_line else 5

        if self.predictions:
            for entry in data:
                for _, models in entry["activations"].items():
                    for name, layer in models.items():
                        if name == "Output":
                            outputs = layer.shape[-1]
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

            outputs = 3 if outputs > 3 else outputs
            height *= outputs

        fig = plt.figure(figsize=(height, 12))

        return fig


    def averages(
        self,
        data: list[dict[str, dict[str, torch.Tensor]]],
    ) -> torch.tensor:
        """Returns averages of all activations for each input.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.

        Returns:
            List of the average activations for each input.
        """
        averages = []
        layer_averages = []
        for entry in data:
            for _, models in entry["activations"].items():
                for name, layer in models.items():
                    if name == "Output":
                        continue
                    layer = layer.squeeze()
                    average = torch.mean(layer)
                    layer_averages.append(average)
                average = sum(layer_averages) / len(layer_averages)
            averages.append(average)

        return averages


    def plot(self, data: list[float]):
        data = np.array([data])