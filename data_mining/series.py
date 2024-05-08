from __future__ import annotations

import pandas as pd
import numpy as np

from data_mining.mining import DataMining

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Series(DataMining):
    """Finds neurons with a series of large or low values.
    
    Attributes:
        data: Panda Dataframe of inputs, model, layer, and neuron activations.
    """
    
    def __init__(self, data: list[dict[str, dict[str, torch.Tensor]]] = None):
        """Initializes data required for mining series of data.

        Args:
            data: File name, model / layer data, and other attributes of all inputs.
        """
        super().__init__(data)
    
    
    def extract(self) -> pd.core.frame.DataFrame:
        model_groups = self.data.groupby(['model', 'layer', 'layer_num'])
        average = model_groups.apply(self.find_extreme)

        print(average)

        a = 1

    def find_extreme(self, activations: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        activations = activations.drop('label', axis = 'columns')
        data_low = activations.apply(lambda x: x.loc[x < 0.2])
        data_high = activations.apply(lambda x: x.loc[x > 0.8])

        data_low.dropna(axis = 'columns', thresh = 3, inplace = True)
        data_high.dropna(axis = 'columns', thresh = 3, inplace = True)

        data_low.apply(self.series)
        data_high.apply(self.series)

        # TODO Work on series data. This can be used on many different inputs to get data.
        # What we're looking for
        #   - columns with a certain number of high or low values
        #   - get indices of all series 
        #       - series is three not nan numbers in a row.

        a = 1

    
    def series(self, neuron: pd.core.Series.series) -> pd.core.Series.series:
        not_null = neuron.notna().astype(int)
        null = neuron.isna().astype(int).cumsum()
        
        series = not_null.groupby(null).sum()

        interesting = ((series > 2) | (series + series.shift(1) > 3)).any()

        if interesting:
            print(neuron)
            print(series)

            series_list = series.to_list()

            indices = self.slice_indices(series_list)

            # TODO Figure out which slices to combine / delete
            # Slice Series and return slices
            # Figure out best format to return slices as
                
            a = 1


    def slice_indices(self, series: list[int]) -> list[tuple[int, int]]:
        start, end = None, None
        slices = []
        index = 0
        first = True

        for value in series:
            if value == 0:
                if start is not None:
                    end = index
                    if end - start > 1:
                        slices.append((start, end))

                if first:
                    index += 1
                    first = False

                start, end = None, None
            else:
                if start is None:
                    start = index
                    index += value - 1
                else:
                    index += value
                
                first = True
            index += 1
        
        if start is not None:
            slices.append((start, -1))

        return slices