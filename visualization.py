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
        visual: Visual = None,
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
            activations = self.normalize(activations)
            
            entry['activations'] = activations
            self.visual.visualize(entry)


    def input_model(self, data: torch.Tensor | tuple, name: str) -> None:
        """Inputs data into the model.

        Args:
            data: data being input into model.
            name: file name used if exception is thrown.

        Raises:
            RuntimeError, TypeError, NameError if input into model was
                unsuccessful.
        """
        try:
            if isinstance(data, tuple):
                _ = self.model(*data)
            else:
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
        dup_models = {}
        names = {}
        activation = {}

        for name, layer in self.model.named_modules():
            if name == "":
                is_model = True
                is_layer = False
            else:
                is_model, is_layer = self.check_layer(layer)

            if is_layer:
                name = str(layer)
                name = name[:name.find('(')]
                
                if name in names:
                    names[name] += 1
                else:
                    names[name] = 1

                name += '_' + str(names[name])

                hook, activation = self.get_activation(name, activation)
                layer.register_forward_hook(hook)
            elif is_model:
                if model_name == "":
                    model_name = str(layer)

                else:
                    model_name = model_name[:model_name.find('(')]
                    
                    if model_name in models:
                        dup_models[model_name] += 1
                        model_name = f"{model_name}_{dup_models[model_name]}"
                    
                    dup_models[model_name] = 1
                    models[model_name] = activation
                    activation = {}
                    names = {}

                    model_name = str(layer)                

        model_name = model_name[:model_name.find('(')]
        if model_name in models:
            dup_models[model_name] += 1
            model_name = f"{model_name}_{dup_models[model_name]}"
        
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

                if output[1] is not None:                
                    if torch.is_tensor(output[1]):
                        activation[name + '_feature2'] = output[1].detach()
                    else:
                        activation[name + '_feature2'] = output[1][0].detach()
                        activation[name + '_feature3'] = output[1][1].detach()
            else:
                activation[name] = output.detach()
        
        return hook, activation
    

    def normalize(
            self,
            activations: dict[str, dict[str, torch.tensor]]
        ) -> dict[str, dict[str, torch.tensor]]:
        """Normalizes network layers, removes models without tensors.

        Args:
            activations: Model layer data.

        Returns:
            activations with normalized layers.
        
        """
        last_key = list(activations)[-1]
        last = list(activations[last_key])[-1]

        activations[last_key]["Output"] = activations[last_key][last]
        del activations[last_key][last]

        activations = {
            key: value
            for key, value in activations.items()
            if len(value) != 0
        }

        for model, layer in activations.items():
            for name, tensor in layer.items():
                tensor -= tensor.min()
                tensor /= tensor.max()
        
        return activations

    
    def check_layer(self, module: torch.nn.Module) -> tuple[bool, bool]:
        """Determines if a module is a model or a layer worth visualizing.

        Args:
            module: Module name used to determine type.

        Returns:
            True, False if it is a model.
            False, True if it is a layer to visualize.
            False, False otherwise.
        """        
        name = str(module)
        name = name[:name.find('(')]
                
        container = (
            name.find('Sequential') != -1 or
            name.find('Module') != -1 or
            name.find('Parameter') != -1 or 
            name.find('Buffer') != -1 or 
            name.find('Dropout') != -1 or
            name.find('Loss') != -1
        )
        
        activation = (
            name.find('ReLU') != -1 or
            name.find('Sigmoid') != -1 or
            name.find('ELU') != -1 or
            name.find('Soft') != -1 or
            name.find('Tanh') != -1 or
            name.find('Hard') != -1 or
            name.find('Norm') != -1 or
            name.find('SiLU') != -1 or
            name.find('Mish') != -1 or
            name.find('GLU') != -1 or
            name.find('Attention') != -1
        )

        other = (
            name.find('Threshold') != -1 or
            name.find('Pad1d') != -1 or
            name.find('Pad2d') != -1 or
            name.find('Pad3d') != -1 or
            name.find('Unsampl') != -1 or
            name.find('huffle') != -1 or
            name.find('DataParallel') != -1 or
            name.find('Identity') != -1 or
            name.find('Embedding') != -1 or
            name.find('Cosine') != -1 or
            name.find('Distance') != -1 or 
            name.find('Unfold') != -1 or 
            name.find('Fold') != -1
        )

        if container or activation or other:
            return False, False
        
        layer = (
            name.find('ool1d') != -1 or
            name.find('ool2d') != -1 or
            name.find('ool3d') != -1 or
            name.find('Conv1d') != -1 or
            name.find('Conv2d') != -1 or
            name.find('Conv3d') != -1 or
            name.find('ConvTranspose') != -1 or
            name.find('ConvLayerBlock') != -1 or
            name.find('TransformerEncoder') != -1 or
            name.find('TransformerDecoder') != -1 or
            isinstance(module, torch.nn.Linear) or
            isinstance(module, torch.nn.Bilinear) or
            isinstance(module, torch.nn.LazyLinear) or
            isinstance(module, torch.nn.Transformer) or
            isinstance(module, torch.nn.RNN) or
            isinstance(module, torch.nn.GRU) or
            isinstance(module, torch.nn.LSTM)
        )

        model = not layer

        return model, layer