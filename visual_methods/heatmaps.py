from __future__ import annotations

import copy
import math
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from visualize import Visual


class Heatmaps(Visual):
    """Visualizes neural network activation layers with a heatmap.

    Generates heatmap for each layer, saves all heatmaps
    onto one figure.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        video: Whether video is wanted for some visualizations.
    """

    def __init__(
        self,
        visual: Visual = None,
        folder: str = "Heatmaps",
        video: bool = False
    ):
        """Intializes visualization methods and folders.

        Args:
            visual: Allows for other forms of visualization.
            folder: Used to save different forms of visualization.
            video: Whether video is wanted for some visualizations.
        """
        super().__init__(visual, folder, video)

    def visualize(self, data: dict[str, dict[str, torch.Tensor]]) -> None:
        """Visualizes selected neural network layers with heatmaps.

        Args:
            data: File name, model / layer data, and other attributes.
        """
        visual_data = copy.deepcopy(data)

        for model, layers in data["activations"].items():
            print(f"\nVisualizing model: {model}")

            layers = self.correct_dimension(layers, data["file"], model)
            subfigs = self.init_plot(data["file"], model, layers)

            i = 0
            for name, activation in layers.items():
                print(f"Visualizing layer: {name}")

                activations = self.activations_2D(activation)

                if not isinstance(activations, list):
                    activations = [activations]

                self.plot(name, activations, subfigs[i])

                i += 1

            self.save(data["file"], model)
        
        self.save_videos(data)

        if self.visual is not None:
            self.visual.visualize(visual_data)

    def correct_dimension(
        self, data: dict[str, dict[str, torch.Tensor]], file: str, model: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Reduces layers in model if there are too many to visualize.

        If there are too many layers present, layers are sliced into
        two different groups.

        Args:
            data: Model / layer data.
            file: File name.
            model: Model name.

        Returns:
            data: Model / layer data sliced into correct dimension.
        """
        total = 0
        index = 0
        new_data = {}

        for name, activation in data.items():
            activations = self.activations_2D(activation)

            if not isinstance(activations, list):
                activations = [activations]

            layer_rows, _ = self.dimensions(activations)
            total += layer_rows

            if total > 14:
                new_layers = {f"{model}_2": dict(list(data.items())[index:])}
                new_data["activations"] = new_layers
                new_data["file"] = file
                data = dict(list(data.items())[:index])

                self.visualize(new_data)
                return data

            index += 1
        
        return data

    def init_plot(
        self, file: str, title: str, layers: dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Initializes the plot for a model.

        Vertically divides the figure into the number of layers present.
        Layers with more data are given more vertical space.

        Args:
            file: Input file.
            title: Title of the plot.
            layers: Activations of layers.

        Returns:
            Subaxes of subplots for plot.
        """
        rows = len(layers)
        columns = 1

        ratios = self.ratios(layers)

        if rows == 1:
            rows = 2
            ratios = [10, 1]

        fig = plt.figure(figsize=(20, 12))

        new_title = f"{title}, File: {file}"
        if len(new_title) > 24:
            new_title = f"{title}\nFile: {file}"

        fig.suptitle(new_title, x=0.06, y=0.995, fontsize="x-large")

        subfigs = fig.subfigures(rows, columns, height_ratios=ratios)

        return subfigs

    def plot(
        self,
        title: str,
        activations: list[np.ndarray],
        subplot: plt.SubFigure,
    ) -> None:
        """Plots all the activations with heatmaps for a layer.

        Args:
            title: Title for subplot.
            activations: All 2D activation maps for a layer.
            subplot: Subplot location to plot layer.
        """
        layer_rows, layer_columns = self.dimensions(activations)

        subaxes = subplot.subplots(layer_rows, layer_columns)

        x = 0.06 if len(title) > 24 else 0.05
        subplot.suptitle(title, x=x, y=0.45)

        if isinstance(subaxes, np.ndarray):
            subaxes = subaxes.flatten()
            for ax in subaxes:
                ax.axis("off")

        i = 0

        for layer in activations:
            if isinstance(subaxes, np.ndarray):
                ax = subaxes[i]
            else:
                ax = subaxes
            ax.axis("off")

            if len(layer.shape) == 1:
                layer = [layer]

            sns.heatmap(layer, cbar=False, ax=ax)

            i += 1

    def dimensions(self, activations: list[np.ndarray]) -> tuple(int, int):
        """Determines the number rows and columns needed to plot activations.

        Args:
            activations: All activation maps for a layer.

        Returns:
            Rows and columns needed to plot every activation map.
        """
        columns = 16 if len(activations) > 16 else len(activations)
        rows = math.ceil(len(activations) / columns)

        if rows <= 0:
            rows = 1

        return rows, columns

    def ratios(self, layers: dict[str, torch.Tensor]) -> list[int]:
        """Determines vertical ratios needed for different layers.

        Layers that require more rows are given more vertical space.

        Args:
            layers: Activations of layers.

        Returns:
            Ratios for how to vertically partion the subplots.
        """
        ratios = []

        for name, activation in layers.items():
            layer_rows, _ = self.dimensions(activation)

            ratios.append(layer_rows)

        return ratios
