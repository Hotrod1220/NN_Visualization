from __future__ import annotations

import torch
import copy
import pandas as pd

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class DataMining(ABC):
    """Abstract class for data mining tasks.

    Attributes:
        data: Panda Dataframe of inputs, model, layer, and neuron activations.
    """
    
    def __init__(
            self,
            data: list[dict[str, dict[str, torch.Tensor]]] = None,
        ):
        """Initializes data required for data mining tasks.

        Hidden layers in recurrent networks are removed if present.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
        """
        self.data = data


    @property
    def data(self):
        """Dataframe data to be mined. Converts list of dictionaries into dataframe."""
        return self._data
    

    @data.setter
    def data(self, obj: list[dict[str, dict[str, torch.Tensor]]]):
        if obj is not None:
            obj = self.remove_hidden(obj)
            self._data = self.dataframe(obj)
        else:
            self._data = obj


    @abstractmethod
    def extract(self) -> pd.core.frame.DataFrame:
        pass

    
    def remove_hidden(
            self,
            data: list[dict[str, dict[str, torch.Tensor]]]
        ) -> list[dict[str, dict[str, torch.Tensor]]]:
        """Removes hidden layers from the recurrent layers if present.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.

        Returns:
            Data with recurrent layers removed.
        """
        new_data = copy.deepcopy(data)
        for i in range(len(data)):
            for model, models in data[i]["activations"].items():
                for name, layer in models.items():
                    hidden = (
                        name.find("_feature2") != -1 or 
                        name.find("_feature3") != -1
                    )
                    if hidden:
                        _ = new_data[i]["activations"][model].pop(name)

        return new_data
    

    def dataframe(
            self,
            data: list[dict[str, dict[str, torch.Tensor]]]
        ) -> pd.core.frame.DataFrame:
        """Converts data from list of dictionaries into a DataFrame.

        Data format:
                           model    layer   layer_num      1         ...     n
           'input_name'    Model1   layer1      1        value11,    ...,  value1n
           'input_name'    Model2   layer2      2        value21,    ...,  value2n
        
        Args:
            data: Data converted into a dataframe.

        Returns:
            Dataframe in format shown above.
        """
        dataframe = pd.DataFrame()
        for entry in data:
            label = entry["name"]
            for model, models in entry["activations"].items():
                for name, layer in models.items():
                    if name != "Output":
                        layers = self.decompose(layer)
                        if not isinstance(layers, list):
                            layers = [layers]
                        
                        for i, activation in enumerate(layers):
                            new_info = pd.DataFrame(
                                [[model, name, i, label]],
                                columns = ["model", "layer", "layer_num", "label"],
                            )        
                            activations = pd.DataFrame([activation])

                            new_info = pd.concat(
                                [new_info, activations],
                                axis = 'columns'
                            )
                            dataframe = pd.concat(
                                [dataframe, new_info],
                                ignore_index = True
                            )

        return dataframe
    

    def decompose(self, array: torch.Tensor | np.ndarray) -> list[np.ndarray]:
        """Decomposes larger dimension tensors into all its 1D arrays.

        Args:
            tensor: tesnor to get arrays from.

        Returns:
            list of 1D numpy arrays contained in larger dimension tensor.
        """
        if isinstance(array, torch.Tensor):
            tensor = array.squeeze()
            array = tensor.detach().cpu().numpy()

        activations = []

        if len(array.shape) < 2:
            return array
            
        for i in range(array.shape[0]):
            activations.append(self.decompose(array[i]))

        return activations
    