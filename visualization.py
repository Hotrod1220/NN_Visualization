from __future__ import annotations

import torch

from visualize import Visual

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any, Callable


class Visualization:
    """Visualization of PyTorch neural networks.

    This class employs the Model-View-Controller (MVC) design pattern with
    dependency injection to offer flexibility in handling different models,
    visualization techniques, and data sources.

    Attributes:
        model: PyTorch model that provides neuron activations.
        data: Contains file name, Model-ready input and other attributes 
            used for visualization.
        visual: Forms of visualization desired.
    """

    def __init__(
        self,
        model: Any,
        model_input: list[dict],
        visual: Visual = None
    ):
        """Intializes model, input for model and the visualization methods.

        Args:
            model: PyTorch model.
            model_input: Contains file name, model-ready input and other attributes
                used for visualization.
            visual: Forms of visualization desired, defaults to None.
        """
        self.model = model
        self.data = model_input
        self.visual = visual


    @property
    def data(self):
        """Data for input into model and visualization methods.
        
        Data must be a list of dictionaries with 'data' and 'file' keys.
        'data' is model-ready input, 'file' is a file string.
        """
        return self._data


    @data.setter
    def data(self, obj):
        try:
            correct_format = all(
                item['data'] is not None and 
                item['file'] is not None
                for item in obj
            )

            if correct_format:
                self._data = obj
            else:
                raise Exception(
                    "'data' or 'file' is None for an item in the model input "
                    "list, initialize before setting."
                )
        except KeyError:
            raise Exception(
                "Input for model must contain a list of dictionaries with "
                "'data' and 'file' keys."
            )


    def visualize(self) -> None:
        """Visualizes neural network with chosen visualization methods.
        
        Raises:
            Exception if self.visual is None.
        """
        if self.visual is None:
            raise Exception(
                "Visualization not specific (self.visual is None), "
                "initialize before calling."
            )
        
        activations = self.activations()
        
        for entry in self.data:
            model_data = entry['data']
            name = entry['file']

            print(f"\nVisualizing file: {name}")

            self.input_model(model_data, name)

            entry['activations'] = activations
            self.visual.visualize(entry)


    def input_model(self, data: torch.Tensor, name: str) -> None:
        """Inputs data into the model.

        Args:
            data: data being input into model.
            name: file name used if exception is thrown.

        Raises:
            RuntimeError, TypeError, NameError if input into model was
                unsuccessful.
        """
        try:
            _ = self.model(data)

        except (RuntimeError, TypeError, NameError) as error:
            raise Exception(
                f"Error inputting file: {name} into model.\n"
                f"{error}"
            )


    def activations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Loads the model layers into hooks. 
        
        Allows for layer activations to be accessed when data is 
        input into model.

        Returns:
            Activations of all layers of all models contained in self.model
                Example of return:
                {
                    'Model1' : {'Layer_1' : Tensor(...), 'Layer_2' : Tensor(...)}
                    'Model2' : {'Layer_1' : Tensor(...), 'Layer_2' : Tensor(...)}
                }
        """
        model_name = ''
        models = {}
        names = {}
        activation = {}

        for name, layer in self.model.named_modules():
            # TODO Check in future if name != '' will cause problems with other models.

            if name != '':
                name = str(layer)
                name = name[:name.find('(')]
                
                not_dropout = name.find('Dropout') == -1
                
                not_container = (
                    name.find('Sequential') == -1 and
                    name.find('Module') == -1 and
                    name.find('Parameter') == -1
                )
                
                not_activation = (
                    name.find('ReLU') == -1 and
                    name.find('Sigmoid') == -1 and
                    name.find('ELU') == -1 and
                    name.find('Softmax') == -1 and
                    name.find('Softmin') == -1 and
                    name.find('Tanh') == -1 and
                    name.find('Hard') == -1 and
                    name.find('Norm') == -1
                )

                if not_dropout and not_container and not_activation:
                    if name in names:
                        names[name] += 1
                    else:
                        names[name] = 1

                    name += '_' + str(names[name])

                    hook, activation = self.get_activation(name, activation)
                    layer.register_forward_hook(hook)
            else:
                if model_name == '':
                    model_name = str(layer)

                else:
                    model_name = model_name[:model_name.find('(')]
                    models[model_name] = activation

                    model_name = str(layer)

        model_name = model_name[:model_name.find('(')]
        models[model_name] = activation

        return models


    def get_activation(
            self,
            name: str,
            activation: dict[str, torch.Tensor]
        ) -> tuple(Callable, dict):
        """Retrieves the activation layer values from a hook.

        Args:
            name: name of the layer to get values.
            activation: activations of different layers.

        Returns:
            Callable function used for activation retrieval on 
                register_forward_hook call. 
            Activations of different layers.
        """
        def hook(model, input, output):
            if not torch.is_tensor(output):
                activation[name] = output[0].detach()
                
                if torch.is_tensor(output[1]):
                    activation[name + '_feature2'] = output[1].detach()
                else:
                    activation[name + '_feature2'] = output[1][0].detach()
                    activation[name + '_feature3'] = output[1][1].detach()
            else:
                activation[name] = output.detach()
        
        return hook, activation
