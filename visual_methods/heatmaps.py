from __future__ import annotations

import os
import copy
import math
import imageio
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from natsort import natsorted 
from visualize import Visual

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    import torch


class Heatmaps(Visual):
    """Visualizes neural network activation layers with a heatmap.

    Generates heatmap for each layer, saves all heatmaps
    onto one figure.

    Attributes:
        visual: Allows for other forms of visualization.
        folder: Used to save different forms of visualization.
        video: Whether a video should be created from the heatmaps created.
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
            video: Whether a video should be created from the heatmaps created.
        """
        super().__init__(visual, folder)
        self._video = video


    def visualize(self, data: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        """Visualizes neural network layers with heatmaps on one plot.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
        """
        visual_data = copy.deepcopy(data)

        for entry in data:
            for model, layers in entry["activations"].items():
                print(f"\nVisualizing: {entry['name']}")
                print(f"\nVisualizing model: {model}")

                layers = self.correct_dimension(layers, entry['name'], model)
                subfigs = self.init_plot(entry['name'], model, layers)

                i = 0
                for name, activation in layers.items():
                    print(f"Visualizing layer: {name}")

                    activations = self.activations_2D(activation)

                    if not isinstance(activations, list):
                        activations = [activations]

                    output = 'labels' in entry and name == 'Output'
                    if output:
                        self.plot(name, activations, subfigs[i], entry['labels'])
                    else:
                        self.plot(name, activations, subfigs[i])

                    i += 1

                    bottom = 0.2 if output else None
                    plt.subplots_adjust(bottom = bottom, right = 1, top = 1)
                
                self.save(entry['name'], model)

        if self._video:
            self.save_videos(data)
        
        if self.visual is not None:
            self.visual.visualize(visual_data)


    def correct_dimension(
        self,
        data: dict[str, dict[str, torch.Tensor]],
        file: str,
        model: str
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
                new_data['name'] = file
                data = dict(list(data.items())[:index])

                self.visualize([new_data])
                return data

            index += 1
                
        return data


    def init_plot(
        self,
        file: str,
        title: str,
        layers: dict[str, torch.Tensor]
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

        fig = plt.figure(figsize = (20, 12))

        new_title = f"{title}, File: {file}"
        fontsize = "x-large"
        if len(new_title) > 24:
            new_title = f"{title}\nFile: {file}"
            fontsize = "large"

        fig.suptitle(new_title, x = 0.06, y = 0.995, fontsize = fontsize)

        subfigs = fig.subfigures(rows, columns, height_ratios = ratios)

        return subfigs


    def plot(
        self,
        title: str,
        activations: list[np.ndarray],
        subplot: plt.SubFigure,
        labels: list[str | int | float] = None
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
        subplot.suptitle(title, x = x, y = 0.55)

        output = title == "Output" and labels is not None

        if isinstance(subaxes, np.ndarray):
            subaxes = subaxes.flatten()
            for ax in subaxes:
                if not output:
                    ax.axis("off")

        i = 0
        for layer in activations:
            if isinstance(subaxes, np.ndarray):
                ax = subaxes[i]
            else:
                ax = subaxes

            if len(layer.shape) == 1:
                layer = [layer]

            if output:
                sns.heatmap(
                    layer,
                    cbar = False,
                    ax = ax,
                    xticklabels = labels,
                    yticklabels = False
                )
            else:
                ax.axis("off")
                sns.heatmap(layer, cbar = False, ax = ax)

            i += 1


    def dimensions(self, activations: list[np.ndarray]) -> tuple[int, int]:
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


    def create_video(self, name: str, path: pathlib.PosixPath | str) -> None:
        """Creates a video from images contained in a folder.
         
        Args:
            name: video name.
            path: folder to get images and save video.
        """
        if isinstance(path, str):
            path = Path(path)

        files = []
        for p in path.rglob("*"):
            f_path = path.joinpath(p)
            if f_path.is_file() and f_path.suffix == '.png':
                files.append(f_path)

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


    def save_videos(self, data: dict[str, dict[str, torch.Tensor]]) -> None:
        """Transforms model heatmaps into a video.

        Args:
            data: File name, model / layer data, and other attributes.
        """
        path = Path.cwd().joinpath("Visualization")
        path = path.joinpath(self.folder)

        entry = data[0]
        models = set()
        for model, _ in entry["activations"].items():
            models.add(model)

        name = data[-1]['name']
        for model in models:
            file = path.joinpath(model)
            self.create_video(name, file)
